# CURE: Confidence-based Uncertainty Regularization Enhancement

## Installation

```bash
pip install torch gym d4rl wandb pyrallis numpy
```

## Usage

### CURE-CQL
```bash
python cure_cql.py --env halfcheetah-medium-v2 --seed 0
```

### CURE-TD3+BC
```bash
python cure_td3bc.py --env halfcheetah-medium-v2 --seed 0
```

### Run baseline (without CURE)
```bash
python cure_cql.py --env halfcheetah-medium-v2 --use_cure False --seed 0
```

## Examples

```bash
# Different environments
python cure_cql.py --env walker2d-medium-replay-v2 --seed 0
python cure_td3bc.py --env hopper-medium-expert-v2 --seed 0

# Custom hyperparameters
python cure_cql.py --env halfcheetah-medium-v2 --n_critics 7 --cure_alpha_lr 1e-4 --seed 0
```

## Supported Environments
All D4RL environments: `halfcheetah`, `hopper`, `walker2d`, `antmaze`, `pen`, `door`, `hammer`, `relocate`
