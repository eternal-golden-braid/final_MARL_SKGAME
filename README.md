# Multi-Robot Battery Disassembly Coordination

This repository contains the code for training and evaluating reinforcement learning agents for multi-robot coordination in a battery disassembly task. The system uses a Stackelberg game approach where a leader robot acts first, followed by two follower robots.

## Overview

In this environment, three robots collaborate to disassemble a battery module:
- **Franka robot (Leader)**: Equipped with a two-finger gripper for unbolting operations
- **UR10 robot (Follower 1)**: Equipped with vacuum suction for sorting and pick-and-place
- **Kuka robot (Follower 2)**: Equipped with specialized tools for casing and connections

The task board contains different types of tasks, each suited to a specific robot or requiring collaboration between multiple robots.

## Improvements

The codebase has been refactored and extended with the following improvements:

### Environment Improvements
1. **Enhanced State Representation**: Flattened board, robot positions, time budget, and action availability mask
2. **Action Masking**: Boolean mask for valid actions, with invalid logits set to -infinity
3. **Reward Shaping**: Improved with step penalty, task-cleared bonus, row bonus, and collision penalty
4. **Proper Success Probabilities**: Now correctly samples from Bernoulli distribution using task-specific probabilities

### Algorithm Improvements
1. **Fixed Network Architecture**: Feature extractors now created in `__init__` for more consistent behavior
2. **QMIX Implementation**: Added value factorization for improved multi-agent coordination
3. **Stackelberg Hierarchy**: Better implementation of leader-follower relationships
4. **Action Space Handling**: Improved mapping between continuous and discrete actions
5. **More Stable Replay Buffer**: Fixed padding for recurrent networks to avoid repeating terminal states

### Implementation Hygiene
1. **Unit Consistency**: Consistent use of success probabilities without hardcoded values
2. **Proper Device Placement**: Allocate tensors on the correct device from the start
3. **Debug Printing**: Guarded by debug flag to avoid slowing down training
4. **Hyper-parameter Management**: More organized parameter handling and configuration options
5. **Comprehensive Evaluation**: More thorough evaluation with statistical significance
6. **Weights & Biases Integration**: Added support for tracking experiments with W&B including model architecture, gradients, and training metrics

## Algorithms

The repository implements four different reinforcement learning approaches:

1. **DRQN (Deep Recurrent Q-Network)**: The baseline implementation that uses recurrent neural networks (LSTM) to handle temporal dependencies
2. **DDQN+PER (Double DQN with Prioritized Experience Replay)**: Reduces overestimation bias and samples more informative experiences
3. **C51 (Categorical 51)**: Distributional RL approach that models the full distribution of returns
4. **SAC (Soft Actor-Critic)**: Policy gradient approach with entropy regularization (not fully implemented in this version)

## Project Structure

```
.
├── env
│   └── battery_disassembly_env.py  # Environment implementation
├── agents
│   ├── base_agent.py               # Base agent class and QMIX network
│   ├── replay_buffer.py            # Experience replay buffer implementations
│   ├── drqn_agent.py               # DRQN agent implementation
│   ├── ddqn_per_agent.py           # DDQN+PER agent implementation
│   └── c51_agent.py                # C51 agent implementation
├── tests
│   └── drqn_regression_test.py     # Regression test for DRQN
├── train.py                        # Main training script
├── README.md                       # This file
└── CHANGELOG.md                    # Detailed list of changes
```

## Usage

### Training

To train an agent, use the `train.py` script:

```bash
python train.py --algorithm drqn --train --n_episodes 1000 --hidden_size 128 --lr 1e-4
```

To use Weights & Biases for experiment tracking:

```bash
python train.py --algorithm drqn --train --n_episodes 1000 --use_wandb --wandb_project "multi-robot-disassembly"
```

Available algorithm options:
- `drqn`: Deep Recurrent Q-Network (baseline)
- `ddqn`: Double DQN with Prioritized Experience Replay
- `c51`: Categorical 51 Distributional RL

### Evaluation

To evaluate a trained agent:

```bash
python train.py --algorithm drqn --eval --n_eval_episodes 50 --exp_name drqn_experiment --render
```

### Regression Testing

To run a regression test to ensure DRQN's performance doesn't degrade:

```bash
python tests/drqn_regression_test.py --n_episodes 20 --checkpoint checkpoints/drqn_baseline/model_final
```

Or using the main script:

```bash
python train.py --baseline drqn
```

## Configuration Options

The training script supports a wide range of configuration options:

### General Parameters
- `--algorithm`: Algorithm to use (drqn, ddqn, c51)
- `--train`: Train the agent
- `--eval`: Evaluate the agent
- `--resume`: Resume training from checkpoint
- `--baseline`: Run regression test against baseline
- `--gpu`: Use GPU if available
- `--render`: Render environment during training/evaluation

### Environment Parameters
- `--task_id`: Task ID (1 for default, 2 for larger 8x6 board)
- `--max_steps`: Maximum steps per episode

### Training Parameters
- `--n_episodes`: Number of episodes
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--gamma`: Discount factor
- `--epsilon`: Initial epsilon for exploration
- `--epsilon_decay`: Epsilon decay rate
- `--epsilon_min`: Minimum epsilon
- `--hidden_size`: Hidden size for networks
- `--buffer_size`: Replay buffer size
- `--sequence_length`: Sequence length for recurrent agents
- `--update_every`: Update target network every N steps
- `--eval_every`: Evaluate every N episodes
- `--n_eval_episodes`: Number of evaluation episodes
- `--seed`: Random seed
- `--use_qmix`: Use QMIX for value factorization

### Algorithm-Specific Parameters
- `--lstm_layers`: Number of LSTM layers (DRQN)
- `--n_atoms`: Number of atoms (C51)
- `--v_min`, `--v_max`: Value bounds (C51)

## Requirements

- Python 3.8+
- PyTorch 2.1+
- NumPy
- Matplotlib
- tensorboardX
- tqdm

## License

[MIT License](LICENSE)