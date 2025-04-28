"""
Regression test for the DRQN algorithm.

This script tests that the DRQN algorithm still performs as expected
after code refactoring.
"""
import os
import sys
import numpy as np
import torch
import argparse
import time
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battery_disassembly_env import BatteryDisassemblyEnv
from agents.drqn_agent import StackelbergDRQNAgent
from agents.replay_buffer import SequenceReplayBuffer


def run_drqn_regression_test(seed: int = 42, n_episodes: int = 20, checkpoint_path: Optional[str] = None,
                           baseline_reward: float = -5.0, render: bool = False) -> bool:
    """
    Run a regression test on the DRQN algorithm.
    
    Args:
        seed: Random seed
        n_episodes: Number of episodes to run
        checkpoint_path: Path to a checkpoint to load
        baseline_reward: Minimum average reward to pass the test
        render: Whether to render the environment
        
    Returns:
        True if the test passes, False otherwise
    """
    print("Running DRQN regression test...")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env_params = {
        'task_id': 1,
        'seed': seed,
        'max_time_steps': 20,  # Short episodes for quick testing
        'debug': False
    }
    env = BatteryDisassemblyEnv(env_params)
    
    # Get environment information
    env_info = env.get_task_info()
    
    # Get initial state to determine state dimension
    _, _, info = env.reset_env()
    if 'enhanced_state' in info:
        # Use the enhanced state dimension
        state_dim = info['enhanced_state'].shape[0]
    else:
        # Use the original state dimension
        state_dim = env_info['dims']
    
    action_dim_leader = env_info['dimAl']
    action_dim_follower1 = env_info['dimAf1']
    action_dim_follower2 = env_info['dimAf2']
    
    # Create agent
    agent = StackelbergDRQNAgent(
        state_dim=state_dim,
        action_dim_leader=action_dim_leader,
        action_dim_follower1=action_dim_follower1,
        action_dim_follower2=action_dim_follower2,
        hidden_size=128,  # Use the default configuration
        sequence_length=8,
        device='cpu',
        learning_rate=1e-4,
        gamma=0.9,
        epsilon=0.01,  # Low exploration for testing
        epsilon_decay=0.995,
        epsilon_min=0.01,
        tau=0.01,
        update_every=10,
        use_qmix=False,
        lstm_layers=1,
        seed=seed
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded agent from {checkpoint_path}")
    
    # Run episodes
    rewards = []
    leader_rewards = []
    follower1_rewards = []
    follower2_rewards = []
    completion_rates = []
    steps_per_episode = []
    
    for episode in range(n_episodes):
        # Reset environment and agent
        first_row, _, info = env.reset_env()
        agent.reset_hidden_states()
        
        episode_reward = 0
        episode_leader_reward = 0
        episode_follower1_reward = 0
        episode_follower2_reward = 0
        steps = 0
        done = False
        
        # Initialize renderer if needed
        if render:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()
        
        # Run episode
        while not done and steps < env_params['max_time_steps']:
            # Use enhanced state if available
            if 'enhanced_state' in info:
                state_to_use = info['enhanced_state']
            else:
                state_to_use = first_row
            
            # Get action masks if available
            action_masks = info.get('action_masks', None)
            
            # Select action (no exploration during evaluation)
            leader_action, follower1_action, follower2_action = agent.act(state_to_use, action_masks, epsilon=0)
            
            # Take action
            next_first_row, _, rl, rf1, rf2, done, next_info = env.step(leader_action, follower1_action, follower2_action)
            
            # Update statistics
            episode_reward += (rl + rf1 + rf2) / 3  # Average reward across agents
            episode_leader_reward += rl
            episode_follower1_reward += rf1
            episode_follower2_reward += rf2
            steps += 1
            
            # Render environment if requested
            if render:
                env.render(ax)
                plt.draw()
                plt.pause(0.1)
            
            # Update state
            first_row = next_first_row
            info = next_info
        
        # Close renderer if open
        if render:
            plt.ioff()
            plt.close()
        
        # Record results
        rewards.append(episode_reward)
        leader_rewards.append(episode_leader_reward)
        follower1_rewards.append(episode_follower1_reward)
        follower2_rewards.append(episode_follower2_reward)
        steps_per_episode.append(steps)
        completion_rates.append(float(done))
        
        print(f"Episode {episode+1}/{n_episodes}: " + 
              f"Reward={episode_reward:.2f}, " +
              f"L={episode_leader_reward:.2f}, " +
              f"F1={episode_follower1_reward:.2f}, " +
              f"F2={episode_follower2_reward:.2f}, " +
              f"Steps={steps}, Done={done}")
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_completion = np.mean(completion_rates)
    mean_steps = np.mean(steps_per_episode)
    std_steps = np.std(steps_per_episode)
    
    # Calculate 95% confidence intervals
    ci_95_reward = 1.96 * std_reward / np.sqrt(n_episodes)
    ci_95_completion = 1.96 * np.std(completion_rates) / np.sqrt(n_episodes)
    ci_95_steps = 1.96 * std_steps / np.sqrt(n_episodes)
    
    print(f"Average reward: {mean_reward:.2f} ± {ci_95_reward:.2f}")
    print(f"Average steps: {mean_steps:.2f} ± {ci_95_steps:.2f}")
    print(f"Completion rate: {mean_completion*100:.1f}% ± {ci_95_completion*100:.1f}%")
    
    # Test passes if:
    # 1. Mean reward >= baseline_reward
    # 2. At least one episode completes
    # 3. No errors during execution
    passed = mean_reward >= baseline_reward and mean_completion > 0
    
    if passed:
        print("DRQN regression test: PASS")
    else:
        print("DRQN regression test: FAIL")
        print(f"Expected reward >= {baseline_reward}, got {mean_reward:.2f}")
        print(f"Expected at least one completed episode, got {mean_completion*100:.1f}%")
    
    return passed


def save_baseline_performance(n_episodes: int = 50, save_path: str = "checkpoints/drqn_baseline"):
    """
    Save DRQN baseline performance for future regression testing.
    
    Args:
        n_episodes: Number of episodes to run
        save_path: Path to save the baseline
    """
    print("Saving DRQN baseline performance...")
    
    # Create environment
    env_params = {
        'task_id': 1,
        'seed': 42,
        'max_time_steps': 100,
        'debug': False
    }
    env = BatteryDisassemblyEnv(env_params)
    
    # Get environment information
    env_info = env.get_task_info()
    
    # Get initial state to determine state dimension
    _, _, info = env.reset_env()
    if 'enhanced_state' in info:
        # Use the enhanced state dimension
        state_dim = info['enhanced_state'].shape[0]
    else:
        # Use the original state dimension
        state_dim = env_info['dims']
    
    action_dim_leader = env_info['dimAl']
    action_dim_follower1 = env_info['dimAf1']
    action_dim_follower2 = env_info['dimAf2']
    
    # Create agent
    agent = StackelbergDRQNAgent(
        state_dim=state_dim,
        action_dim_leader=action_dim_leader,
        action_dim_follower1=action_dim_follower1,
        action_dim_follower2=action_dim_follower2,
        hidden_size=128,
        sequence_length=8,
        device='cpu',
        learning_rate=1e-4,
        gamma=0.9,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        tau=0.01,
        update_every=10,
        use_qmix=False,
        lstm_layers=1,
        seed=42
    )
    
    # Create replay buffer
    buffer_size = 10000
    sequence_length = 8
    batch_size = 32
    buffer = SequenceReplayBuffer(buffer_size, sequence_length, batch_size, seed=42)
    
    # Training loop
    training_stats = {
        'rewards': [],
        'completion_rates': [],
        'steps': []
    }
    
    for episode in range(n_episodes):
        # Reset environment and agent
        first_row, _, info = env.reset_env()
        agent.reset_hidden_states()
        
        episode_reward = 0
        steps = 0
        done = False
        
        # Run episode
        while not done and steps < env_params['max_time_steps']:
            # Use enhanced state if available
            if 'enhanced_state' in info:
                state_to_use = info['enhanced_state']
            else:
                state_to_use = first_row
            
            # Get action masks if available
            action_masks = info.get('action_masks', None)
            
            # Select action
            leader_action, follower1_action, follower2_action = agent.act(state_to_use, action_masks)
            
            # Take action
            next_first_row, _, rl, rf1, rf2, done, next_info = env.step(leader_action, follower1_action, follower2_action)
            
            # Get next state to use
            if 'enhanced_state' in next_info:
                next_state_to_use = next_info['enhanced_state']
            else:
                next_state_to_use = next_first_row
            
            # Add to replay buffer
            experience = (state_to_use, leader_action, follower1_action, follower2_action, 
                        rl, rf1, rf2, next_state_to_use, done)
            buffer.add(experience)
            
            # Update statistics
            episode_reward += (rl + rf1 + rf2) / 3  # Average reward across agents
            steps += 1
            
            # Update agent if enough samples in buffer
            if len(buffer) >= batch_size:
                samples = buffer.sample(batch_size)
                agent.update(samples)
            
            # Update state
            first_row = next_first_row
            info = next_info
        
        # End episode in buffer
        buffer.end_episode(zero_state=info.get('enhanced_state', None))
        
        # Update training statistics
        training_stats['rewards'].append(episode_reward)
        training_stats['completion_rates'].append(float(done))
        training_stats['steps'].append(steps)
        
        print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Steps={steps}, Done={done}")
    
    # Save agent and stats
    os.makedirs(save_path, exist_ok=True)
    agent.save(os.path.join(save_path, "model_final"))
    
    import pickle
    with open(os.path.join(save_path, "training_stats.pkl"), "wb") as f:
        pickle.dump(training_stats, f)
    
    # Get final performance
    mean_reward = np.mean(training_stats['rewards'][-10:])  # Last 10 episodes
    mean_completion = np.mean(training_stats['completion_rates'][-10:])
    
    print(f"Baseline performance (last 10 episodes):")
    print(f"Average reward: {mean_reward:.2f}")
    print(f"Completion rate: {mean_completion*100:.1f}%")
    
    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DRQN regression test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_episodes', type=int, default=20, help='Number of episodes to run')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--baseline_reward', type=float, default=-5.0, help='Minimum average reward to pass the test')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save_baseline', action='store_true', help='Save baseline performance')
    
    args = parser.parse_args()
    
    if args.save_baseline:
        save_baseline_performance(n_episodes=50)
    else:
        success = run_drqn_regression_test(
            seed=args.seed,
            n_episodes=args.n_episodes,
            checkpoint_path=args.checkpoint,
            baseline_reward=args.baseline_reward,
            render=args.render
        )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)