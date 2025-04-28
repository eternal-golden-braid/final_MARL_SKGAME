# CHANGELOG

## Version 1.1.0

### Major Changes

#### Logging Improvements
- Added Weights & Biases (wandb) integration for comprehensive experiment tracking
- Added model architecture logging to wandb
- Added gradient and parameter tracking for neural networks
- Added detailed per-episode statistics for evaluation runs
- Added support for wandb projects and team/entity organization

## Version 1.0.0

### Major Changes

#### Environment Improvements
- **Enhanced State Representation** - Added `get_enhanced_state()` method that provides a flattened representation of the full board, robot positions, time budget, and action masks in a single vector
- **Action Masking** - Added `get_action_masks()` method that returns boolean masks for valid actions for each robot
- **Reward Shaping** - Added step penalty, task-cleared bonus, row bonus, and collision penalty for more informative rewards
- **Success Probabilities** - Fixed probability sampling to correctly use Bernoulli distribution with task-specific probabilities

#### Algorithm Fixes and Improvements
- **Feature Extractor Creation** - Moved feature extractor creation to `__init__` in all network classes to avoid rebuilding on every forward pass
- **QMIX Implementation** - Added `QMIXNetwork` class for value factorization and monotonic value function approximation
- **Stackelberg Hierarchy** - Improved leader-follower dynamics in agent update methods to properly enforce Stackelberg equilibrium
- **Discrete-Continuous Action Mapping** - Fixed bias in continuous to discrete action mapping
- **Replay Buffer Padding** - Fixed sequence padding to avoid repeating terminal transitions

#### Implementation Hygiene
- **Unit Consistency** - Removed hardcoded 0.7/0.9 success probabilities in favor of task-specific values
- **Device Placement** - Fixed device placement by allocating tensors on the correct device from the start
- **Debug Prints** - Added `debug` flag and guarded print statements to avoid slowing down training
- **Training Protocol** - Added wall-clock time tracking and more comprehensive statistics

### Specific Issue Fixes

1. **State non-Markov & too small** - FIXED 
   - Implemented enhanced state in `get_enhanced_state()` that includes full board state, robot positions, time budget
   - Added this information to the environment state dictionary

2. **Action space lacks validity mask** - FIXED
   - Implemented `get_action_masks()` method to provide boolean masks for valid actions
   - Added `apply_action_mask()` in base agent to apply masks to Q-values

3. **Reward sparse & symmetric** - FIXED
   - Added reward shaping with step penalty, task-cleared bonus, row bonus, and collision penalty
   - Balanced rewards for different agents and task types

4. **Stackelberg hierarchy not enforced** - FIXED
   - Redesigned `compute_stackelberg_equilibrium()` method to properly compute Stackelberg equilibrium
   - Leader now properly takes into account follower's best responses

5. **Critic rebuilds its feature extractor each forward pass** - FIXED
   - Moved feature extractor creation to `__init__` in all network classes
   - Fixed RecurrentQNetwork and modified C51Network accordingly

6. **Continuous→discrete mapping bias** - FIXED
   - Modified action space mapping functions for proper conversion between continuous and discrete spaces
   - Added checks to handle edge cases and invalid actions

7. **Replay padding repeats terminal transition** - FIXED
   - Modified `end_episode()` in replay buffer to use zero-state padding instead of repeating terminal states
   - Added `zero_state` parameter to better handle padding

8. **Extraneous prints slow training** - FIXED
   - Added `debug` flag to control print statements
   - Guarded all print statements with debug checks

9. **IndexError on joint-action tensor slice** - FIXED
   - Fixed tensor dimensionality issues in all agent implementations
   - Added proper reshaping and broadcasting for tensor operations

10. **Evaluation uses only 5 episodes → no CIs** - FIXED
    - Increased default evaluation episodes to 50
    - Added calculation of 95% confidence intervals for evaluation metrics
    - Added statistical reporting to evaluation results

### New Features

- **Modular Architecture** - Refactored code into modular components for better maintainability
- **QMIX Network** - Added value function factorization for more stable multi-agent learning
- **Comprehensive Logging** - Added TensorBoard integration for better visualization of training progress
- **Regression Testing** - Added dedicated regression test to ensure DRQN performance doesn't degrade
- **Experiment Configuration** - Added YAML configuration support and improved command-line arguments
- **Larger Task Board** - Added support for 8x6 task boards (task_id=2)

### Code Organization

- Created modular directory structure with separate files for:
  - Environment (`env/battery_disassembly_env.py`)
  - Agents (`agents/*.py`)
  - Testing (`tests/drqn_regression_test.py`)
  - Training (`train.py`)

### DRQN-Specific Changes

- Fixed hidden state handling to maintain temporal dependencies
- Added proper sequence handling in the replay buffer
- Fixed tensor dimensionality issues in LSTM processing
- Ensured DRQN regression test passes with minimal changes to core functionality

## Known Limitations and Future Work

- SAC implementation is not fully complete and requires additional work
- Graph-based observation representation not fully implemented
- Curriculum learning framework outlined but not fully implemented
- Self-play and learning-guided search are left as TODOs for future work