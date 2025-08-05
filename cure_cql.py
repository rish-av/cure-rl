# CURE: Confidence-based Uncertainty Regularization Enhancement for CQL
# Based on bootstrap ensemble uncertainty estimation
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    device: str = "cuda"
    env: str = "walker2d-medium-replay-v2"
    seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99

    # SAC actor and alpha params
    backup_entropy: bool = False
    use_automatic_entropy_tuning: bool = True
    policy_lr: float = 3e-5
    qf_lr: float = 3e-4
    alpha_lr: float = 3e-4
    soft_target_update_rate: float = 5e-3
    target_update_period: int = 1

    # CQL specific parameters
    cql_n_actions: int = 10
    cql_temp: float = 1.0
    cql_importance_sample: bool = True
    cql_lagrange: bool = False
    cql_target_action_gap: float = -1.0
    cql_max_target_backup: bool = False
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf

    # CURE parameters
    use_cure: bool = True
    cure_target_penalty: float = 15.0
    cure_alpha_lr: float = 3e-4
    cure_warmup_steps: int = 0

    # Fixed CQL alpha (used if use_cure=False or during warm-up)
    cql_alpha: float = 5.0

    # General network parameters
    n_critics: int = 5
    orthogonal_init: bool = True
    q_n_hidden_layers: int = 3
    bc_steps: int = int(0)
    bc_loss_weight: float = 0.0

    # Normalization
    normalize: bool = True
    normalize_reward: bool = False
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    # CURE bounds
    min_alpha: float = 0.1
    max_alpha: float = 15.0

    # Project info
    project: str = "CURE"
    group: str = "CURE-D4RL"
    name: str = "CURE-CQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# --- UTILITY FUNCTIONS ---
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std
    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
        save_code=True,
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


# --- DATA HANDLING ---
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


# --- NETWORK ARCHITECTURES ---
def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False):
        super().__init__()
        self.log_std_min, self.log_std_max, self.no_tanh = log_std_min, log_std_max, no_tanh

    def log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1)) if not self.no_tanh else Normal(mean, std)
        return torch.sum(dist.log_prob(sample), dim=-1)

    def forward(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1)) if not self.no_tanh else Normal(mean, std)
        sample = torch.tanh(mean) if deterministic and not self.no_tanh else (mean if deterministic else dist.rsample())
        log_prob = torch.sum(dist.log_prob(sample), dim=-1)
        return sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, orthogonal_init: bool = False, no_tanh: bool = False):
        super().__init__()
        self.action_dim, self.max_action, self.no_tanh = action_dim, max_action, no_tanh
        self.base_network = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2 * action_dim))
        if orthogonal_init: 
            self.base_network.apply(lambda m: init_module_weights(m, orthogonal_init))
        self.log_std_multiplier = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.log_std_offset = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3: 
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        mean, log_std = torch.split(self.base_network(observations), self.action_dim, dim=-1)
        return self.tanh_gaussian.log_prob(mean, self.log_std_multiplier * log_std + self.log_std_offset, actions)

    def forward(self, observations: torch.Tensor, deterministic: bool = False, repeat: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None: 
            observations = extend_and_repeat(observations, 1, repeat)
        mean, log_std = torch.split(self.base_network(observations), self.action_dim, dim=-1)
        actions, log_probs = self.tanh_gaussian(mean, self.log_std_multiplier * log_std + self.log_std_offset, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False, n_hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(state_dim + action_dim, 256), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, orthogonal_init))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Handle multiple actions case
        multiple_actions = False
        batch_size = state.shape[0]
        if action.ndim == 3 and state.ndim == 2:
            multiple_actions = True
            state = extend_and_repeat(state, 1, action.shape[1]).reshape(-1, state.shape[-1])
            action = action.reshape(-1, action.shape[-1])
        
        input_tensor = torch.cat([state, action], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        
        return q_values


class AdaptiveAlpha(nn.Module):
    def __init__(self, state_dim: int, uncertainty_dim: int = 1, 
                 orthogonal_init: bool = True, 
                 min_alpha: float = 0.1, max_alpha: float = 10.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim + uncertainty_dim),
            nn.Linear(state_dim + uncertainty_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, orthogonal_init))

    def forward(self, state_uncertainty: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.network(state_uncertainty))
        return self.min_alpha + (raw * (self.max_alpha - self.min_alpha))
    
    def compute_calibrated_target_alpha(self, uncertainty: torch.Tensor, base_alpha: float = None) -> torch.Tensor:
        """Compute uncertainty-calibrated target alpha values"""
        if base_alpha is None:
            base_alpha = self.min_alpha
        uncertainty_90th = torch.quantile(uncertainty.squeeze(-1), 0.9)
        relative_uncertainty = (uncertainty.squeeze(-1) - uncertainty_90th)/ (uncertainty_90th + 1e-6)
        target_alpha = base_alpha + (self.max_alpha - self.min_alpha) * torch.sigmoid(relative_uncertainty - 1)
        return target_alpha.clamp(self.min_alpha, self.max_alpha)


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


# --- Main Trainer Class ---
class ContinuousCURE:
    def __init__(self, config: TrainConfig, actor: TanhGaussianPolicy, critics: nn.ModuleList, critic_optimizers: List[torch.optim.Optimizer], cure_alpha_net: Optional[AdaptiveAlpha] = None):
        self.config = config
        self.device = config.device
        self.total_it = 0
        
        self.actor = actor
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.policy_lr)
        
        self.critics = critics
        self.critic_optimizers = critic_optimizers
        self.target_critics = deepcopy(self.critics).to(self.device)
        
        self.target_entropy = -np.prod(actor.action_dim)
        if config.use_automatic_entropy_tuning:
            self.log_sac_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.sac_alpha_optimizer = torch.optim.AdamW([self.log_sac_alpha], lr=config.alpha_lr)

        # Add Lagrange multiplier for CQL
        if config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0).to(self.device)
            self.alpha_prime_optimizer = torch.optim.AdamW(self.log_alpha_prime.parameters(), lr=config.qf_lr)
        else:
            self.log_alpha_prime = None

        self.cure_alpha_net = cure_alpha_net
        if self.config.use_cure:
            self.cure_alpha_optimizer = torch.optim.AdamW(self.cure_alpha_net.parameters(), lr=config.cure_alpha_lr)
            print(f"Using CURE with target penalty: {config.cure_target_penalty}")
        else:
            print(f"Using fixed-alpha CQL with value: {config.cql_alpha}")

        print("Bootstrap sampling enabled for independent critics.")
        self.bootstrap_indices = [None] * self.config.n_critics

    def _get_bootstrap_batch(self, full_batch: List[torch.Tensor], critic_idx: int) -> List[torch.Tensor]:
        """Creates a bootstrapped batch for a given critic."""
        if self.total_it % 5000 == 0 or self.bootstrap_indices[critic_idx] is None:
            batch_size = full_batch[0].shape[0]
            self.bootstrap_indices[critic_idx] = torch.randint(
                0, batch_size, (batch_size,), device=self.device
            )
        
        indices = self.bootstrap_indices[critic_idx]
        bootstrapped_batch = [tensor[indices] for tensor in full_batch]
        return bootstrapped_batch

    def update_target_network(self):
        tau = self.config.soft_target_update_rate
        for critic, target_critic in zip(self.critics, self.target_critics):
            soft_update(target_critic, critic, tau)

    def _compute_cql_loss(self, observations: torch.Tensor, actions: torch.Tensor, 
                         next_observations: torch.Tensor, q_pred: torch.Tensor, 
                         critic_idx: int = None) -> Tuple[torch.Tensor, Dict]:
        """Compute the complete CQL loss"""
        batch_size = observations.shape[0]
        action_dim = actions.shape[-1]
        
        # Sample random actions
        cql_random_actions = actions.new_empty(
            (batch_size, self.config.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        
        # Sample current policy actions
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.config.cql_n_actions
        )
        
        # Sample next state policy actions
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.config.cql_n_actions
        )
        
        # Detach actions to prevent policy gradients in critic update
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )
        
        # Compute Q-values for all action types
        if critic_idx is not None:
            # Single critic case
            critic = self.critics[critic_idx]
            cql_q_rand = critic(observations, cql_random_actions)
            cql_q_current_actions = critic(observations, cql_current_actions)
            cql_q_next_actions = critic(observations, cql_next_actions)
        else:
            # Multiple critics case - use minimum
            cql_q_rand = torch.min(torch.stack([
                critic(observations, cql_random_actions) for critic in self.critics
            ]), dim=0)[0]
            cql_q_current_actions = torch.min(torch.stack([
                critic(observations, cql_current_actions) for critic in self.critics
            ]), dim=0)[0]
            cql_q_next_actions = torch.min(torch.stack([
                critic(observations, cql_next_actions) for critic in self.critics
            ]), dim=0)[0]
        
        # Build CQL cat tensor
        if self.config.cql_importance_sample:
            # Apply importance sampling corrections
            random_density = np.log(0.5**action_dim)
            cql_cat_q = torch.cat(
                [
                    cql_q_rand - random_density,
                    cql_q_next_actions - cql_next_log_pis.detach(),
                    cql_q_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
        else:
            # Standard CQL without importance sampling
            cql_cat_q = torch.cat(
                [
                    cql_q_rand,
                    torch.unsqueeze(q_pred, 1),
                    cql_q_next_actions,
                    cql_q_current_actions,
                ],
                dim=1,
            )
        
        # Compute logsumexp
        cql_qf_ood = torch.logsumexp(cql_cat_q / self.config.cql_temp, dim=1) * self.config.cql_temp
        
        # Compute CQL difference and apply clipping
        cql_qf_diff = torch.clamp(
            cql_qf_ood - q_pred,
            self.config.cql_clip_diff_min,
            self.config.cql_clip_diff_max,
        )
        
        return cql_qf_diff, {
            'cql_q_rand': cql_q_rand.mean().item(),
            'cql_q_current_actions': cql_q_current_actions.mean().item(),
            'cql_q_next_actions': cql_q_next_actions.mean().item(),
            'cql_qf_ood': cql_qf_ood.mean().item(),
            'cql_qf_diff': cql_qf_diff.mean().item(),
        }

    def train(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        log_dict = {}

        # Pre-calculate uncertainty for CURE
        with torch.no_grad():
            q_preds = [critic(observations, actions) for critic in self.critics]
            uncertainty = torch.std(torch.stack(q_preds), dim=0)

        is_cure_active = self.config.use_cure and self.total_it >= self.config.cure_warmup_steps
        cure_alpha = None
        if is_cure_active:
            state_uncertainty = torch.cat([observations, uncertainty.unsqueeze(-1)], dim=1)
            cure_alpha = self.cure_alpha_net(state_uncertainty).squeeze(-1)

        # --- Critic Update Logic ---
        all_td_losses, all_cql_losses = [], []
        
        for i in range(self.config.n_critics):
            obs_i, act_i, rew_i, next_obs_i, done_i = self._get_bootstrap_batch(batch, i)

            # TD Target calculation
            with torch.no_grad():
                next_actions_i, next_log_pi_i = self.actor(next_obs_i)
                
                if self.config.cql_max_target_backup:
                    # Use max target backup
                    next_actions_expanded, _ = self.actor(next_obs_i, repeat=self.config.cql_n_actions)
                    target_q_preds_i = torch.stack([tc(next_obs_i, next_actions_expanded) for tc in self.target_critics])
                    min_target_q_i, max_target_indices = torch.max(torch.min(target_q_preds_i, dim=0)[0], dim=-1)
                    # Update next_log_pi for the selected actions
                    next_log_pi_i = torch.gather(next_log_pi_i.repeat(1, self.config.cql_n_actions).view(next_obs_i.shape[0], -1), 
                                               -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
                else:
                    # Standard target backup
                    target_q_preds_i = torch.stack([tc(next_obs_i, next_actions_i) for tc in self.target_critics])
                    min_target_q_i = torch.min(target_q_preds_i, dim=0)[0].squeeze(-1)

                if self.config.backup_entropy:
                    sac_alpha_val = self.log_sac_alpha.exp().detach()
                    min_target_q_i -= sac_alpha_val * next_log_pi_i
                
                td_target_i = rew_i.squeeze(-1) + (1.0 - done_i.squeeze(-1)) * self.config.discount * min_target_q_i
            
            # TD Loss
            q_pred_i = self.critics[i](obs_i, act_i).squeeze(-1)
            td_loss_i = F.mse_loss(q_pred_i, td_target_i)

            # CQL Loss with complete implementation
            cql_diff_i, cql_metrics = self._compute_cql_loss(obs_i, act_i, next_obs_i, q_pred_i, critic_idx=i)
            
            # Apply regularization
            if is_cure_active:
                cure_alpha_i = cure_alpha[self.bootstrap_indices[i]]
                cql_loss_i = (cure_alpha_i.detach() * cql_diff_i).mean()
            elif self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_loss_i = alpha_prime * self.config.cql_alpha * (cql_diff_i - self.config.cql_target_action_gap)
            else:
                cql_loss_i = self.config.cql_alpha * cql_diff_i.mean()
            
            # Final critic loss
            critic_loss_i = td_loss_i + cql_loss_i
            self.critic_optimizers[i].zero_grad()
            critic_loss_i.backward()
            self.critic_optimizers[i].step()

            all_td_losses.append(td_loss_i.item())
            all_cql_losses.append(cql_loss_i.item())
            
            # Update log dict with individual critic metrics
            log_dict.update({
                f"critic_{i+1}_td_loss": td_loss_i.item(),
                f"critic_{i+1}_cql_loss": cql_loss_i.item(),
                f"critic_{i+1}_total_loss": critic_loss_i.item(),
                f"critic_{i+1}_q_mean": q_pred_i.mean().item(),
                **{f"critic_{i+1}_{k}": v for k, v in cql_metrics.items()}
            })
        
        td_loss = sum(all_td_losses) / self.config.n_critics
        cql_loss = sum(all_cql_losses) / self.config.n_critics
        critic_loss = td_loss + cql_loss

        # --- CQL Lagrange Alpha Prime Update ---
        if self.config.cql_lagrange and not is_cure_active:
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
            avg_cql_diff = sum(all_cql_losses) / len(all_cql_losses) / (alpha_prime * self.config.cql_alpha)
            alpha_prime_loss = -alpha_prime * self.config.cql_alpha * (avg_cql_diff.detach() - self.config.cql_target_action_gap)
            alpha_prime_loss.backward()
            self.alpha_prime_optimizer.step()
            log_dict.update({
                "alpha_prime_loss": alpha_prime_loss.item(),
                "alpha_prime": alpha_prime.item(),
            })

        # === CURE Alpha Update ===
        if is_cure_active:
            target_alpha = self.cure_alpha_net.compute_calibrated_target_alpha(
                uncertainty, base_alpha=self.config.min_alpha
            )
            
            calibration_loss = F.mse_loss(cure_alpha, target_alpha.detach())
            lagrangian_weight = 0.1
            
            # Compute average CQL diff for Lagrangian component
            avg_cql_diff_cure = torch.tensor(sum(all_cql_losses) / len(all_cql_losses), device=self.device)
            lagrangian_loss = lagrangian_weight * (cure_alpha * (self.config.cure_target_penalty - avg_cql_diff_cure.detach())).mean()
            
            cure_alpha_loss = calibration_loss + lagrangian_loss
            
            self.cure_alpha_optimizer.zero_grad()
            cure_alpha_loss.backward()
            self.cure_alpha_optimizer.step()
            
            log_dict.update({
                "cure_alpha_loss": cure_alpha_loss.item(),
                "cure_calibration_loss": calibration_loss.item(),
                "cure_lagrangian_loss": lagrangian_loss.item(),
                "cure_alpha_mean": cure_alpha.mean().item(),
                "cure_alpha_std": cure_alpha.std().item(),
                "cure_target_alpha_mean": target_alpha.mean().item(),
                "cure_uncertainty_mean": uncertainty.mean().item(),
            })
        
        # --- Actor and SAC Alpha Update ---
        new_actions, log_pi = self.actor(observations)
        q_new_actions = [critic(observations, new_actions) for critic in self.critics]
        min_q_new = torch.min(torch.stack(q_new_actions), dim=0)[0]
        
        sac_alpha_val = self.log_sac_alpha.exp().detach() if self.config.use_automatic_entropy_tuning else 0.0
        rl_actor_loss = (sac_alpha_val * log_pi - min_q_new).mean()

        if self.total_it < self.config.bc_steps:
            policy_mean, _ = self.actor.base_network(observations).chunk(2, dim=-1)
            bc_actions = torch.tanh(policy_mean) * self.actor.max_action
            bc_loss = F.mse_loss(bc_actions, actions)
            actor_loss = rl_actor_loss + self.config.bc_loss_weight * bc_loss
            log_dict["actor_bc_loss"] = bc_loss.item()
        else:
            actor_loss = rl_actor_loss
            log_dict["actor_bc_loss"] = 0.0
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_sac_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.sac_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.sac_alpha_optimizer.step()
            log_dict["sac_alpha_loss"] = alpha_loss.item()
            log_dict["sac_alpha_value"] = self.log_sac_alpha.exp().item()

        if self.total_it % self.config.target_update_period == 0:
            self.update_target_network()

        # Core logging
        log_dict.update({
            "critic_td_loss": td_loss,
            "critic_cql_loss": cql_loss,
            "critic_total_loss": critic_loss,
            "actor_total_loss": actor_loss.item(),
            "actor_rl_loss": rl_actor_loss.item(),
        })
        
        return log_dict


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    set_seed(config.seed, env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward or "antmaze" in config.env:
        modify_reward(dataset, config.env, config.reward_scale, config.reward_bias)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # Create replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    # Create networks
    actor = TanhGaussianPolicy(state_dim, action_dim, max_action, config.orthogonal_init).to(config.device)
    
    critics = nn.ModuleList([
        FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init, config.q_n_hidden_layers).to(config.device)
        for _ in range(config.n_critics)
    ])
    
    critic_optimizers = [torch.optim.AdamW(critic.parameters(), lr=config.qf_lr) for critic in critics]

    cure_alpha_net = AdaptiveAlpha(state_dim, orthogonal_init=config.orthogonal_init,
                                min_alpha=config.min_alpha, max_alpha=config.max_alpha).to(config.device) if config.use_cure else None
    
    trainer = ContinuousCURE(config, actor, critics, critic_optimizers, cure_alpha_net)
    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    wandb_init(asdict(config))
    
    print("---------------------------------------")
    print(f"Training {config.name}, Env: {config.env}, Seed: {config.seed}, N_Critics: {config.n_critics}")
    print(f"CQL Features - Importance Sampling: {config.cql_importance_sample}, Lagrange: {config.cql_lagrange}, Max Target: {config.cql_max_target_backup}")
    print("---------------------------------------")

    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            eval_scores = eval_actor(env, actor, device=config.device, n_episodes=config.n_episodes, seed=config.seed)
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            
            print("---------------------------------------")
            print(f"Time steps: {t + 1}/{int(config.max_timesteps)} | "
                  f"Evaluation over {config.n_episodes} episodes: "
                  f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}")
            print("---------------------------------------")
            wandb.log({"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it)

            if config.use_cure and trainer.cure_alpha_net is not None:
                with torch.no_grad():
                    sample_states = batch[0][:5]
                    sample_actions = batch[1][:5]
                    sample_uncertainty = torch.std(torch.stack([critic(sample_states, sample_actions) for critic in trainer.critics]), dim=0)
                    state_uncertainty = torch.cat([sample_states, sample_uncertainty.unsqueeze(-1)], dim=1)
                    sample_alphas = trainer.cure_alpha_net(state_uncertainty)
                    print(f"Sample uncertainties: {sample_uncertainty.squeeze().cpu().numpy()}")
                    print(f"Sample adaptive alphas: {sample_alphas.squeeze().cpu().numpy()}")

            if (t + 1) % 50000 == 0:
                checkpoint_path = f"checkpoints/{config.env}_{config.seed}_checkpoint_{t + 1}.pt" if config.checkpoints_path is None else os.path.join(config.checkpoints_path, f"checkpoint_{t + 1}.pt")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dicts": [critic.state_dict() for critic in critics],
                    "critic_optimizers_state_dicts": [opt.state_dict() for opt in critic_optimizers],
                    "cure_alpha_net_state_dict": trainer.cure_alpha_net.state_dict() if trainer.cure_alpha_net else None,
                    "total_it": trainer.total_it,
                }, checkpoint_path)

if __name__ == "__main__":
    train()