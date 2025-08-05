# CURE: Confidence-based Uncertainty Regularization Enhancement for TD3+BC
# Clean implementation for submission
import copy
import os
import random
import uuid
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

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Basic TD3+BC parameters
    project: str = "CORL"
    group: str = "TD3_BC_CURE-D4RL"
    name: str = "TD3_BC_CURE"
    env: str = "walker2d-medium-replay-v2"
    alpha: float = 5
    discount: float = 0.99
    expl_noise: float = 0.1
    tau: float = 5e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    max_timesteps: int = int(1e6)
    buffer_size: int = 2_000_000
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    seed: int = 0
    device: str = "cuda"
    
    # CURE-specific parameters
    use_cure: bool = True
    
    # Ensemble parameters for uncertainty estimation
    n_critics: int = 3
    
    # State-adaptive parameters
    cure_alpha_lr: float = 1e-5
    cure_warmup_steps: int = 0
    
    # Network parameters
    orthogonal_init: bool = True
    q_n_hidden_layers: int = 3

    min_alpha: float = 0.1
    max_alpha: float = 5.0

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# Utility Functions
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
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std
    
    def scale_reward(reward):
        return reward_scale * reward
    
    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
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
def eval_actor(env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int) -> np.ndarray:
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

def return_reward_range(dataset, max_episode_steps):
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

def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

# Network Architectures
def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, orthogonal_init: bool = False):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        
        if orthogonal_init:
            self.net.apply(lambda m: init_module_weights(m, orthogonal_init))
        
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False, n_hidden_layers: int = 3):
        super(Critic, self).__init__()
        
        layers = [nn.Linear(state_dim + action_dim, 256), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        
        self.net = nn.Sequential(*layers)
        
        if orthogonal_init:
            self.net.apply(lambda m: init_module_weights(m, orthogonal_init))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


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

    def forward(self, state: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        state_uncertainty = torch.cat([state, uncertainty], dim=1)
        raw = torch.sigmoid(self.network(state_uncertainty))
        return self.min_alpha + (raw * (self.max_alpha - self.min_alpha))
    
    def compute_calibrated_target_alpha(self, uncertainty: torch.Tensor, base_alpha: float = None) -> torch.Tensor:
        """Compute uncertainty-calibrated target alpha values"""
        if base_alpha is None:
            base_alpha = self.min_alpha
        uncertainty_90th = torch.quantile(uncertainty.squeeze(-1), 0.9)
        relative_uncertainty = (uncertainty.squeeze(-1) - uncertainty_90th) / (uncertainty_90th + 1e-6)
        target_alpha = base_alpha + (self.max_alpha - self.min_alpha) * torch.sigmoid(relative_uncertainty - 1)
        return target_alpha.clamp(self.min_alpha, self.max_alpha)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
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


# CURE Main Implementation with Independent Critics
class TD3_BC_CURE:
    def __init__(
        self,
        config: TrainConfig,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critics: List[nn.Module],
        critics_optimizers: List[torch.optim.Optimizer],
        adaptive_alpha: Optional[AdaptiveAlpha] = None,
    ):
        self.config = config
        self.device = config.device
        
        # Core networks
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        
        # Ensemble of critics for uncertainty estimation
        self.critics = critics
        self.critics_targets = [copy.deepcopy(critic) for critic in critics]
        self.critics_optimizers = critics_optimizers
        
        # CURE components
        self.adaptive_alpha = adaptive_alpha
        
        if self.adaptive_alpha and config.use_cure:
            self.alpha_optimizer = torch.optim.Adam(
                self.adaptive_alpha.parameters(), 
                lr=config.cure_alpha_lr
            )
            print(f"CURE enabled with {len(critics)} critics ensemble")
        
        # Bootstrap sampling for independent critics
        self.bootstrap_indices = [None] * len(critics)
        self.bootstrap_resample_freq = 5000
        print(f"Independent critics enabled with bootstrap sampling")
        
        # TD3+BC parameters
        self.max_action = float(actor.max_action)
        self.discount = config.discount
        self.tau = config.tau
        self.policy_noise = config.policy_noise * self.max_action
        self.noise_clip = config.noise_clip * self.max_action
        self.policy_freq = config.policy_freq
        self.alpha = config.alpha

        self.total_it = 0

    def get_bootstrap_batch(self, batch: TensorBatch, critic_idx: int) -> TensorBatch:
        """Get bootstrap sample for independent critic training."""
        states, actions, rewards, next_states, dones = batch
        batch_size = len(states)
        
        # Resample bootstrap indices periodically
        if self.total_it % self.bootstrap_resample_freq == 0 or self.bootstrap_indices[critic_idx] is None:
            self.bootstrap_indices[critic_idx] = torch.randint(0, batch_size, (batch_size,), device=self.device)
        
        indices = self.bootstrap_indices[critic_idx]
        return [
            states[indices],
            actions[indices],
            rewards[indices],
            next_states[indices],
            dones[indices]
        ]

    def compute_ensemble_uncertainty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """CURE: Compute epistemic uncertainty from ensemble critic disagreement."""
        q_values = []
        for critic in self.critics:
            q_val = critic(states, actions)
            q_values.append(q_val)
        
        # Stack and compute standard deviation across ensemble
        q_stack = torch.stack(q_values, dim=0)  # [n_critics, batch_size, 1]
        q_std = torch.std(q_stack, dim=0)  # [batch_size, 1]
        
        return q_std

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        # --- Critics Ensemble Update ---
        critics_loss = 0.0
        
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critics_optimizers)):
            # Use bootstrap sampling for independent critics
            boot_batch = self.get_bootstrap_batch(batch, i)
            boot_states, boot_actions, boot_rewards, boot_next_states, boot_dones = boot_batch
            boot_not_done = 1 - boot_dones
            
            with torch.no_grad():
                noise = (torch.randn_like(boot_actions) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = (self.actor_target(boot_next_states) + noise).clamp(
                    -self.max_action, self.max_action
                )
                
                # Use ensemble for target computation
                target_q_values = []
                for critic_target in self.critics_targets:
                    target_q = critic_target(boot_next_states, next_action)
                    target_q_values.append(target_q)
                
                # Use minimum over ensemble for conservative estimation
                target_q = torch.min(torch.stack(target_q_values), dim=0)[0]
                target_q = boot_rewards + boot_not_done * self.discount * target_q
            
            current_q = critic(boot_states, boot_actions)
            bellman_loss = F.mse_loss(current_q, target_q)
            
            optimizer.zero_grad()
            bellman_loss.backward()
            optimizer.step()
            
            critics_loss += bellman_loss.item()
            log_dict[f"critic_{i+1}_loss"] = bellman_loss.item()
        
        log_dict["critics_avg_loss"] = critics_loss / len(self.critics)

        # --- Actor Update with CURE ---
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critics[0](state, pi)
            
            is_cure_active = self.config.use_cure and self.total_it >= self.config.cure_warmup_steps
            
            if is_cure_active and self.adaptive_alpha is not None:
                uncertainty = self.compute_ensemble_uncertainty(state, action)
                alpha_coeffs = self.adaptive_alpha(state, uncertainty).squeeze(-1)
                
                q_loss = -q.mean()
                bc_loss = F.mse_loss(pi, action, reduction='none').mean(dim=1)
                weighted_bc_loss = (alpha_coeffs * bc_loss).mean()
                actor_loss = q_loss + weighted_bc_loss
                
                log_dict["adaptive_alpha_mean"] = alpha_coeffs.mean().item()
                log_dict["adaptive_alpha_std"] = alpha_coeffs.std().item()
                log_dict["adaptive_alpha_min"] = alpha_coeffs.min().item()
                log_dict["adaptive_alpha_max"] = alpha_coeffs.max().item()
                log_dict["uncertainty_mean"] = uncertainty.mean().item()
                log_dict["uncertainty_std"] = uncertainty.std().item()
                log_dict["weighted_bc_loss"] = weighted_bc_loss.item()
            else:
                lmbda = self.alpha / q.abs().mean().detach()
                actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
                log_dict["lmbda"] = lmbda.item()
            
            log_dict["actor_loss"] = actor_loss.item()
            log_dict["q_mean"] = q.mean().item()
            log_dict["bc_mse"] = F.mse_loss(pi, action).item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # === CURE Alpha Update ===
            if is_cure_active and self.adaptive_alpha is not None:
                uncertainty = self.compute_ensemble_uncertainty(state, action)
                predicted_alpha = self.adaptive_alpha(state, uncertainty).squeeze(-1)
                
                # Compute uncertainty-calibrated target alpha
                target_alpha = self.adaptive_alpha.compute_calibrated_target_alpha(
                    uncertainty, 
                    base_alpha=self.config.alpha
                )
                
                # Calibration loss: MSE between predicted and target alpha
                calibration_loss = F.mse_loss(predicted_alpha, target_alpha.detach())
                
                # Lagrangian component for constraint satisfaction
                lagrangian_weight = 0.1
                ideal_bc_strength = self.config.alpha  # Target BC strength
                lagrangian_loss = lagrangian_weight * F.mse_loss(
                    predicted_alpha.mean(), 
                    torch.tensor(ideal_bc_strength, device=predicted_alpha.device, dtype=torch.float32)
                )
                
                # Combined objective
                alpha_loss = calibration_loss + lagrangian_loss
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                # Logging
                log_dict.update({
                    "alpha_loss": alpha_loss.item(),
                    "alpha_calibration_loss": calibration_loss.item(),
                    "alpha_lagrangian_loss": lagrangian_loss.item(),
                    "target_alpha_mean": target_alpha.mean().item(),
                    "target_alpha_std": target_alpha.std().item(),
                    "uncertainty_90th_percentile": torch.quantile(uncertainty.squeeze(-1), 0.9).item(),
                    "alpha_above_min_percent": (predicted_alpha > (self.adaptive_alpha.min_alpha + 0.01)).float().mean().item() * 100,
                })
            
            # Target network updates
            for critic, critic_target in zip(self.critics, self.critics_targets):
                soft_update(critic_target, critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }
        
        # Save all critics
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critics_optimizers)):
            state_dict[f"critic_{i}"] = critic.state_dict()
            state_dict[f"critic_{i}_optimizer"] = optimizer.state_dict()
        
        # Save bootstrap indices
        for i, indices in enumerate(self.bootstrap_indices):
            if indices is not None:
                state_dict[f"bootstrap_indices_{i}"] = indices
        
        if self.adaptive_alpha is not None:
            state_dict.update({
                "adaptive_alpha": self.adaptive_alpha.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            })
            
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        # Load all critics
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critics_optimizers)):
            critic.load_state_dict(state_dict[f"critic_{i}"])
            optimizer.load_state_dict(state_dict[f"critic_{i}_optimizer"])
        
        # Recreate target networks
        self.critics_targets = [copy.deepcopy(critic) for critic in self.critics]

        self.total_it = state_dict["total_it"]
        
        # Load bootstrap indices if available
        for i in range(len(self.bootstrap_indices)):
            key = f"bootstrap_indices_{i}"
            if key in state_dict:
                self.bootstrap_indices[i] = state_dict[key]
        
        if self.adaptive_alpha is not None and "adaptive_alpha" in state_dict:
            self.adaptive_alpha.load_state_dict(state_dict["adaptive_alpha"])
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])


# Training Function
@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    set_seed(config.seed, env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Load and process dataset
    dataset = d4rl.qlearning_dataset(env)
    
    if config.normalize_reward:
        modify_reward(dataset, config.env)
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
        dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
        env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    else:
        state_mean, state_std = 0, 1

    # Create replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    # Initialize networks
    actor = Actor(state_dim, action_dim, max_action, config.orthogonal_init).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    
    # Initialize ensemble of critics
    critics = []
    critics_optimizers = []
    for i in range(config.n_critics):
        critic = Critic(state_dim, action_dim, config.orthogonal_init, config.q_n_hidden_layers).to(config.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4, weight_decay=1e-5)
        critics.append(critic)
        critics_optimizers.append(critic_optimizer)
    
    # Initialize adaptive alpha network if CURE is enabled
    adaptive_alpha = None
    if config.use_cure:
        adaptive_alpha = AdaptiveAlpha(
            state_dim, 
            min_alpha=config.min_alpha, 
            max_alpha=config.max_alpha,
            orthogonal_init=config.orthogonal_init
        ).to(config.device)

    # Initialize trainer
    trainer = TD3_BC_CURE(
        config=config,
        actor=actor,
        actor_optimizer=actor_optimizer,
        critics=critics,
        critics_optimizers=critics_optimizers,
        adaptive_alpha=adaptive_alpha,
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))

    wandb_init(asdict(config))
    
    print("---------------------------------------")
    cure_status = f"with CURE ({config.n_critics} critics)" if config.use_cure else "without CURE"
    print(f"Training TD3+BC {cure_status}")
    print(f"Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    # Training loop
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        
        # Evaluation
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env, actor, device=config.device, n_episodes=config.n_episodes, seed=config.seed
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            
            # CURE debugging info
            if config.use_cure and trainer.adaptive_alpha is not None and t >= config.cure_warmup_steps:
                with torch.no_grad():
                    sample_states = batch[0][:5]
                    sample_actions = batch[1][:5]
                    sample_uncertainty = trainer.compute_ensemble_uncertainty(sample_states, sample_actions)
                    sample_alphas = trainer.adaptive_alpha(sample_states, sample_uncertainty)
                    print(f"Sample uncertainties: {sample_uncertainty.squeeze().cpu().numpy()}")
                    print(f"Sample adaptive alphas: {sample_alphas.squeeze().cpu().numpy()}")
            
            print("---------------------------------------")

            if (t + 1) % 100_000 == 0:
                if config.checkpoints_path is None:
                    checkpoint_path = f"checkpoints/{config.env}_{config.seed}_checkpoint_{t + 1}.pt"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                else:
                    checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{t + 1}.pt")

                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dicts": [critic.state_dict() for critic in critics],
                    "critic_optimizers_state_dicts": [opt.state_dict() for opt in critics_optimizers],
                    "cure_alpha_net_state_dict": trainer.adaptive_alpha.state_dict() if trainer.adaptive_alpha else None,
                    "total_it": trainer.total_it,
                }, checkpoint_path)

            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score},
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()