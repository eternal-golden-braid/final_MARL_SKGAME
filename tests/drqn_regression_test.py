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
from typing import Dict, Tuple, List, Optional, Union, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battery_disassembly_env import BatteryDisassemblyEnv
from agents.drqn_agent import StackelbergDRQNAgent


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
    completion_rates = []
    
    for episode in range(n_episodes):
        # Reset environment and agent
        state, _, info = env.reset_env()
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
                state_to_use = state
            
            # Get action masks if available
            action_masks = info.get('action_masks', None)
            
            # Select action (no exploration during evaluation)
            leader_action, follower1_action, follower2_action = agent.act(state_to_use, action_masks, epsilon=0)
            
            # Take action
            next_state, _, rl, rf1, rf2, done, info = env.step(leader_action, follower1_action, follower2_action)
            
            # Update statistics
            episode_reward += (rl + rf1 + rf2) / 3  # Average reward across agents
            steps += 1
            
            # Update state
            state = next_state
        
        rewards.append(episode_reward)
        completion_rates.append(float(done))
        
        print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Steps={steps}, Done={done}")
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_completion = np.mean(completion_rates)
    
    print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Completion rate: {mean_completion*100:.1f}%")
    
    # Test passes if mean reward >= baseline_reward and at least one episode completes
    passed = mean_reward >= baseline_reward and mean_completion > 0
    
    if passed:
        print("DRQN regression test: PASS")
    else:
        print("DRQN regression test: FAIL")
        print(f"Expected reward >= {baseline_reward}, got {mean_reward:.2f}")
        print(f"Expected at least one completed episode, got {mean_completion*100:.1f}%")
    
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DRQN regression test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_episodes', type=int, default=20, help='Number of episodes to run')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--baseline_reward', type=float, default=-5.0, help='Minimum average reward to pass the test')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    
    args = parser.parse_args()
    
    success = run_drqn_regression_test(
        seed=args.seed,
        n_episodes=args.n_episodes,
        checkpoint_path=args.checkpoint,
        baseline_reward=args.baseline_reward,
        render=args.render
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)