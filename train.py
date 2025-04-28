"""
Training script for multi-robot disassembly coordination.

This script provides the main training loop for different agent types.
"""
import os
import argparse
import numpy as np
import torch
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Union, Any
import tensorboardX as tbx
import wandb

from env.battery_disassembly_env import BatteryDisassemblyEnv
from agents.replay_buffer import SequenceReplayBuffer, PrioritizedSequenceReplayBuffer
from agents.drqn_agent import StackelbergDRQNAgent
from agents.ddqn_per_agent import StackelbergDDQNPERAgent
from agents.c51_agent import StackelbergC51Agent


class SeedConfig:
    """Centralized random seed configuration."""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.np_seed = seed
        self.torch_seed = seed
        
        # Set seeds
        np.random.seed(self.np_seed)
        torch.manual_seed(self.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.torch_seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-robot disassembly coordination')
    
    # General parameters
    parser.add_argument('--algorithm', type=str, default='drqn', choices=['drqn', 'ddqn', 'c51', 'sac'],
                       help='Algorithm to use: drqn, ddqn, c51, sac')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--baseline', type=str, default=None, choices=['drqn', 'ddqn', 'c51', 'sac'],
                       help='Run regression test against baseline')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--render', action='store_true', help='Render environment')
    
    # Environment parameters
    parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for networks')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--sequence_length', type=int, default=8, help='Sequence length for recurrent agents')
    parser.add_argument('--update_every', type=int, default=10, help='Update target network every N steps')
    parser.add_argument('--eval_every', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--n_eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_qmix', action='store_true', help='Use QMIX for value factorization')
    
    # DRQN parameters
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    
    # C51 parameters
    parser.add_argument('--n_atoms', type=int, default=51, help='Number of atoms for C51')
    parser.add_argument('--v_min', type=float, default=-10, help='Minimum value for C51')
    parser.add_argument('--v_max', type=float, default=10, help='Maximum value for C51')
    
    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='logs', help='Logging directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--wandb_project', type=str, default='multi-robot-disassembly', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    return parser.parse_args()


def create_agent(args, env):
    """
    Create an agent based on the specified algorithm.
    
    Args:
        args: Command line arguments
        env: Environment instance
        
    Returns:
        Agent instance
    """
    # Get environment information
    env_info = env.get_task_info()
    state_dim = env_info['dims']
    action_dim_leader = env_info['dimAl']
    action_dim_follower1 = env_info['dimAf1']
    action_dim_follower2 = env_info['dimAf2']
    
    # Determine device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Create agent based on algorithm
    if args.algorithm == 'drqn':
        agent = StackelbergDRQNAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            device=device,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            tau=0.01,
            update_every=args.update_every,
            use_qmix=args.use_qmix,
            lstm_layers=args.lstm_layers,
            seed=args.seed,
            debug=args.debug
        )
    elif args.algorithm == 'ddqn':
        agent = StackelbergDDQNPERAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=args.hidden_size,
            device=device,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            tau=0.01,
            update_every=args.update_every,
            alpha=0.6,
            beta=0.4,
            use_qmix=args.use_qmix,
            seed=args.seed,
            debug=args.debug
        )
    elif args.algorithm == 'c51':
        agent = StackelbergC51Agent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=args.hidden_size,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            device=device,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            tau=0.01,
            update_every=args.update_every,
            seed=args.seed,
            debug=args.debug
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    return agent


def create_buffer(args, env):
    """
    Create a replay buffer based on the specified algorithm.
    
    Args:
        args: Command line arguments
        env: Environment instance
        
    Returns:
        Replay buffer instance
    """
    # Get environment information
    env_info = env.get_task_info()
    state_dim = env_info['dims']
    
    # Create buffer based on algorithm
    if args.algorithm == 'ddqn':
        buffer = PrioritizedSequenceReplayBuffer(
            buffer_size=args.buffer_size,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            alpha=0.6,
            beta=0.4,
            seed=args.seed
        )
    else:
        buffer = SequenceReplayBuffer(
            buffer_size=args.buffer_size,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            seed=args.seed
        )
    
    return buffer


def train(args):
    """
    Train an agent on the battery disassembly task.
    
    Args:
        args: Command line arguments
        
    Returns:
        Trained agent
    """
    # Create experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm}_{args.hidden_size}_{args.lr}_{int(time.time())}"
    
    # Create directories
    log_dir = os.path.join(args.log_dir, args.exp_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up logging
    writer = tbx.SummaryWriter(log_dir)
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb_config = {
            "algorithm": args.algorithm,
            "task_id": args.task_id,
            "hidden_size": args.hidden_size,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "sequence_length": args.sequence_length,
            "use_qmix": args.use_qmix,
            "seed": args.seed,
        }
        
        # Add algorithm-specific parameters
        if args.algorithm == 'drqn':
            wandb_config.update({"lstm_layers": args.lstm_layers})
        elif args.algorithm == 'c51':
            wandb_config.update({
                "n_atoms": args.n_atoms,
                "v_min": args.v_min,
                "v_max": args.v_max
            })
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=wandb_config,
            dir=log_dir
        )
    
    # Set up environment
    env_params = {
        'task_id': args.task_id,
        'seed': args.seed,
        'max_time_steps': args.max_steps,
        'debug': args.debug
    }
    env = BatteryDisassemblyEnv(env_params)
    
    # Create agent and replay buffer
    agent = create_agent(args, env)
    buffer = create_buffer(args, env)
    
    # Resume training if requested
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            agent.load(os.path.join(checkpoint_dir, latest_checkpoint))
            
            # Load training stats if available
            stats_path = os.path.join(checkpoint_dir, f"stats_{latest_checkpoint}.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    training_stats = pickle.load(f)
                start_episode = training_stats.get('episode', 0) + 1
                print(f"Resuming from episode {start_episode}")
            else:
                start_episode = 0
        else:
            print("No checkpoint found, starting from scratch")
            start_episode = 0
    else:
        start_episode = 0
    
    # Initialize training statistics
    training_stats = {
        'episode': start_episode - 1,
        'rewards': [],
        'leader_rewards': [],
        'follower1_rewards': [],
        'follower2_rewards': [],
        'steps': [],
        'completion_rates': [],
        'eval_rewards': [],
        'eval_steps': [],
        'eval_completion_rates': [],
        'wall_time': []
    }
    
    # Log model architecture to wandb if enabled
    if args.use_wandb and hasattr(agent, 'leader_online'):
        # Log model architecture as a string
        leader_model_str = str(agent.leader_online)
        follower1_model_str = str(agent.follower1_online)
        follower2_model_str = str(agent.follower2_online)
        
        wandb.run.summary["leader_model"] = leader_model_str
        wandb.run.summary["follower1_model"] = follower1_model_str
        wandb.run.summary["follower2_model"] = follower2_model_str
        
        # Log model graph if possible
        try:
            dummy_input = torch.zeros(1, args.sequence_length, env.get_task_info()['dims']).to(agent.device)
            if args.algorithm == 'drqn':
                wandb.watch(agent.leader_online, log="all", log_freq=args.eval_every)
                wandb.watch(agent.follower1_online, log="all", log_freq=args.eval_every)
                wandb.watch(agent.follower2_online, log="all", log_freq=args.eval_every)
        except Exception as e:
            print(f"Failed to log model graph to wandb: {e}")
    
    # Training loop
    start_time = time.time()
    for episode in range(start_episode, args.n_episodes):
        # Reset environment and agent
        state, _, info = env.reset_env()
        agent.reset_hidden_states() if hasattr(agent, 'reset_hidden_states') else None
        
        # Initialize episode statistics
        episode_reward = 0
        episode_leader_reward = 0
        episode_follower1_reward = 0
        episode_follower2_reward = 0
        steps = 0
        
        # Render environment if requested (and only for some episodes)
        render = args.render and (episode % 100 == 0)
        if render:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()
        
        # Episode loop
        for step in range(args.max_steps):
            # Use enhanced state if available
            if 'enhanced_state' in info:
                state_to_use = info['enhanced_state']
            else:
                state_to_use = state
            
            # Get action masks if available
            action_masks = info.get('action_masks', None)
            
            # Select action
            leader_action, follower1_action, follower2_action = agent.act(state_to_use, action_masks)
            
            # Take action
            next_state, _, rl, rf1, rf2, done, info = env.step(leader_action, follower1_action, follower2_action)
            
            # Add to replay buffer
            experience = (state_to_use, leader_action, follower1_action, follower2_action, 
                         rl, rf1, rf2, info.get('enhanced_state', next_state), done)
            buffer.add(experience)
            
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
            
            # Update agent if enough samples in buffer
            if len(buffer) >= args.batch_size:
                if args.algorithm == 'ddqn':
                    # Sample with priorities for DDQN
                    samples, indices, weights = buffer.sample(args.batch_size)
                    td_errors, leader_loss, follower1_loss, follower2_loss = agent.update((samples, indices, weights))
                    buffer.update_priorities(indices, td_errors)
                    
                    # Log losses to wandb if enabled
                    if args.use_wandb:
                        wandb.log({
                            "train/leader_loss": leader_loss,
                            "train/follower1_loss": follower1_loss,
                            "train/follower2_loss": follower2_loss,
                            "train/epsilon": agent.epsilon
                        }, step=episode * args.max_steps + step)
                else:
                    # Sample normally for other algorithms
                    samples = buffer.sample(args.batch_size)
                    losses = agent.update(samples)
                    
                    # Log losses to wandb if enabled and returned
                    if args.use_wandb and isinstance(losses, tuple) and len(losses) == 3:
                        leader_loss, follower1_loss, follower2_loss = losses
                        wandb.log({
                            "train/leader_loss": leader_loss,
                            "train/follower1_loss": follower1_loss,
                            "train/follower2_loss": follower2_loss,
                            "train/epsilon": agent.epsilon
                        }, step=episode * args.max_steps + step)
            
            # Check if episode is done
            if done:
                break
            
            # Update state
            state = next_state
        
        # End episode in buffer
        buffer.end_episode()
        
        # Close rendering if active
        if render:
            plt.ioff()
            plt.close()
        
        # Update training statistics
        training_stats['episode'] = episode
        training_stats['rewards'].append(episode_reward)
        training_stats['leader_rewards'].append(episode_leader_reward)
        training_stats['follower1_rewards'].append(episode_follower1_reward)
        training_stats['follower2_rewards'].append(episode_follower2_reward)
        training_stats['steps'].append(steps)
        training_stats['completion_rates'].append(float(done))
        training_stats['wall_time'].append(time.time() - start_time)
        
        # Log to TensorBoard
        writer.add_scalar('train/reward', episode_reward, episode)
        writer.add_scalar('train/leader_reward', episode_leader_reward, episode)
        writer.add_scalar('train/follower1_reward', episode_follower1_reward, episode)
        writer.add_scalar('train/follower2_reward', episode_follower2_reward, episode)
        writer.add_scalar('train/steps', steps, episode)
        writer.add_scalar('train/completion_rate', float(done), episode)
        writer.add_scalar('train/epsilon', agent.epsilon, episode)
        
        # Log to Weights & Biases if enabled
        if args.use_wandb:
            wandb.log({
                "train/reward": episode_reward,
                "train/leader_reward": episode_leader_reward,
                "train/follower1_reward": episode_follower1_reward,
                "train/follower2_reward": episode_follower2_reward,
                "train/steps": steps,
                "train/completion_rate": float(done),
                "train/epsilon": agent.epsilon,
                "train/episode": episode,
                "train/wall_time": time.time() - start_time
            })
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{args.n_episodes}: " +
                  f"Reward={episode_reward:.2f}, " +
                  f"L={episode_leader_reward:.2f}, " +
                  f"F1={episode_follower1_reward:.2f}, " +
                  f"F2={episode_follower2_reward:.2f}, " +
                  f"Steps={steps}, Done={done}, " +
                  f"Epsilon={agent.epsilon:.4f}")
        
        # Evaluate agent periodically
        if episode % args.eval_every == 0:
            eval_stats = evaluate(args, agent, env, args.n_eval_episodes, render=False)
            
            # Update training statistics with evaluation results
            training_stats['eval_rewards'].append(eval_stats['reward'])
            training_stats['eval_steps'].append(eval_stats['steps'])
            training_stats['eval_completion_rates'].append(eval_stats['completion_rate'])
            
            # Log evaluation results to TensorBoard
            writer.add_scalar('eval/reward', eval_stats['reward'], episode)
            writer.add_scalar('eval/steps', eval_stats['steps'], episode)
            writer.add_scalar('eval/completion_rate', eval_stats['completion_rate'], episode)
            
            # Log evaluation results to Weights & Biases if enabled
            if args.use_wandb:
                wandb.log({
                    "eval/reward": eval_stats['reward'],
                    "eval/reward_std": eval_stats['reward_std'],
                    "eval/leader_reward": eval_stats['leader_reward'],
                    "eval/follower1_reward": eval_stats['follower1_reward'],
                    "eval/follower2_reward": eval_stats['follower2_reward'],
                    "eval/steps": eval_stats['steps'],
                    "eval/steps_std": eval_stats['steps_std'],
                    "eval/completion_rate": eval_stats['completion_rate'],
                    "eval/episode": episode
                })
            
            print(f"Evaluation: " +
                  f"Reward={eval_stats['reward']:.2f} ± {eval_stats['reward_std']:.2f}, " +
                  f"Steps={eval_stats['steps']:.2f} ± {eval_stats['steps_std']:.2f}, " +
                  f"Completion={eval_stats['completion_rate']*100:.1f}%")
            
            # Save agent
            agent.save(os.path.join(checkpoint_dir, f"model_{episode}"))
            
            # Save training statistics
            with open(os.path.join(checkpoint_dir, f"stats_model_{episode}.pkl"), 'wb') as f:
                pickle.dump(training_stats, f)
    
    # Save final model
    agent.save(os.path.join(checkpoint_dir, "model_final"))
    
    # Save final training statistics
    with open(os.path.join(checkpoint_dir, "stats_final.pkl"), 'wb') as f:
        pickle.dump(training_stats, f)
    
    # Close TensorBoard writer
    writer.close()
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return agent, training_stats


def evaluate(args, agent, env, n_episodes, render=False):
    """
    Evaluate an agent on the battery disassembly task.
    
    Args:
        args: Command line arguments
        agent: Agent to evaluate
        env: Environment instance
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Evaluation statistics
    """
    # Initialize evaluation statistics
    eval_rewards = []
    eval_leader_rewards = []
    eval_follower1_rewards = []
    eval_follower2_rewards = []
    eval_steps = []
    eval_completion = []
    
    # Evaluation loop
    for episode in range(n_episodes):
        # Reset environment and agent
        state, _, info = env.reset_env()
        agent.reset_hidden_states() if hasattr(agent, 'reset_hidden_states') else None
        
        # Initialize episode statistics
        episode_reward = 0
        episode_leader_reward = 0
        episode_follower1_reward = 0
        episode_follower2_reward = 0
        steps = 0
        
        # Render environment if requested
        if render:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()
        
        # Episode loop
        for step in range(args.max_steps):
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
            episode_leader_reward += rl
            episode_follower1_reward += rf1
            episode_follower2_reward += rf2
            steps += 1
            
            # Render environment if requested
            if render:
                env.render(ax)
                plt.draw()
                plt.pause(0.1)
            
            # Check if episode is done
            if done:
                break
            
            # Update state
            state = next_state
        
        # Close rendering if active
        if render:
            plt.ioff()
            plt.close()
        
        # Update evaluation statistics
        eval_rewards.append(episode_reward)
        eval_leader_rewards.append(episode_leader_reward)
        eval_follower1_rewards.append(episode_follower1_reward)
        eval_follower2_rewards.append(episode_follower2_reward)
        eval_steps.append(steps)
        eval_completion.append(float(done))
        
        # Log individual episode results to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "eval_episode/reward": episode_reward,
                "eval_episode/leader_reward": episode_leader_reward,
                "eval_episode/follower1_reward": episode_follower1_reward,
                "eval_episode/follower2_reward": episode_follower2_reward,
                "eval_episode/steps": steps,
                "eval_episode/completion": float(done),
                "eval_episode/episode": episode
            })
    
    # Calculate statistics
    eval_reward_mean = np.mean(eval_rewards)
    eval_reward_std = np.std(eval_rewards)
    eval_leader_reward_mean = np.mean(eval_leader_rewards)
    eval_leader_reward_std = np.std(eval_leader_rewards)
    eval_follower1_reward_mean = np.mean(eval_follower1_rewards)
    eval_follower1_reward_std = np.std(eval_follower1_rewards)
    eval_follower2_reward_mean = np.mean(eval_follower2_rewards)
    eval_follower2_reward_std = np.std(eval_follower2_rewards)
    eval_steps_mean = np.mean(eval_steps)
    eval_steps_std = np.std(eval_steps)
    eval_completion_rate = np.mean(eval_completion)
    
    # Calculate 95% confidence intervals
    ci_95_reward = 1.96 * eval_reward_std / np.sqrt(n_episodes)
    ci_95_steps = 1.96 * eval_steps_std / np.sqrt(n_episodes)
    ci_95_completion = 1.96 * np.std(eval_completion) / np.sqrt(n_episodes)
    
    # Return evaluation statistics
    return {
        'reward': eval_reward_mean,
        'reward_std': eval_reward_std,
        'reward_ci95': ci_95_reward,
        'leader_reward': eval_leader_reward_mean,
        'leader_reward_std': eval_leader_reward_std,
        'follower1_reward': eval_follower1_reward_mean,
        'follower1_reward_std': eval_follower1_reward_std,
        'follower2_reward': eval_follower2_reward_mean,
        'follower2_reward_std': eval_follower2_reward_std,
        'steps': eval_steps_mean,
        'steps_std': eval_steps_std,
        'steps_ci95': ci_95_steps,
        'completion_rate': eval_completion_rate,
        'completion_rate_ci95': ci_95_completion
    }


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Name of the latest checkpoint, or None if no checkpoints found
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_') and not f.endswith('.pkl')]
    if not checkpoints:
        return None
    
    # Extract episode numbers
    episodes = [int(c.split('_')[1]) for c in checkpoints if c.split('_')[1].isdigit()]
    if not episodes:
        return None
    
    # Find the latest episode
    latest_episode = max(episodes)
    return f"model_{latest_episode}"


def run_regression_test(args):
    """
    Run a regression test against a baseline algorithm.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if test passes, False otherwise
    """
    if args.baseline is None:
        print("No baseline specified for regression test")
        return False
    
    # Set up environment
    env_params = {
        'task_id': args.task_id,
        'seed': args.seed,
        'max_time_steps': args.max_steps,
        'debug': args.debug
    }
    env = BatteryDisassemblyEnv(env_params)
    
    # Create agent for the baseline
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.algorithm = args.baseline
    baseline_agent = create_agent(baseline_args, env)
    
    # Load baseline model if available
    baseline_dir = os.path.join(args.checkpoint_dir, f"{args.baseline}_baseline")
    if os.path.exists(baseline_dir):
        baseline_model = find_latest_checkpoint(baseline_dir)
        if baseline_model:
            baseline_agent.load(os.path.join(baseline_dir, baseline_model))
        else:
            print(f"No baseline model found in {baseline_dir}")
            return False
    else:
        print(f"Baseline directory {baseline_dir} not found")
        return False
    
    # Create agent for the current algorithm
    current_agent = create_agent(args, env)
    
    # Evaluate both agents
    print(f"Running regression test: {args.algorithm} vs {args.baseline}")
    
    # Run short evaluation for the baseline
    baseline_stats = evaluate(args, baseline_agent, env, 20, render=False)
    
    # Run short evaluation for the current algorithm
    current_stats = evaluate(args, current_agent, env, 20, render=False)
    
    # Compare results
    baseline_reward = baseline_stats['reward']
    current_reward = current_stats['reward']
    baseline_completion = baseline_stats['completion_rate']
    current_completion = current_stats['completion_rate']
    
    print(f"Baseline ({args.baseline}): Reward={baseline_reward:.2f}, Completion={baseline_completion*100:.1f}%")
    print(f"Current ({args.algorithm}): Reward={current_reward:.2f}, Completion={current_completion*100:.1f}%")
    
    # Test passes if current reward is at least 95% of baseline reward
    reward_pass = current_reward >= 0.95 * baseline_reward
    completion_pass = current_completion >= 0.95 * baseline_completion
    
    if reward_pass and completion_pass:
        print("Regression test PASSED")
        return True
    else:
        print("Regression test FAILED")
        return False


def visualize_training_stats(stats, save_path=None):
    """
    Visualize training statistics.
    
    Args:
        stats: Dictionary containing training statistics
        save_path: Path to save the figure, or None to display
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot rewards
    axes[0, 0].plot(stats['rewards'], label='Average')
    axes[0, 0].plot(stats['leader_rewards'], label='Leader')
    axes[0, 0].plot(stats['follower1_rewards'], label='Follower1')
    axes[0, 0].plot(stats['follower2_rewards'], label='Follower2')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Rewards per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot steps
    axes[0, 1].plot(stats['steps'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].grid(True)
    
    # Plot completion rate (moving average)
    window_size = min(50, len(stats['completion_rates']))
    if window_size > 0:
        completion_rates = np.array(stats['completion_rates'])
        moving_avg = np.convolve(completion_rates, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(moving_avg * 100)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Completion Rate (%)')
        axes[1, 0].set_title(f'Task Completion Rate (Moving Avg, Window={window_size})')
        axes[1, 0].grid(True)
    
    # Plot evaluation metrics vs training episodes
    if 'eval_rewards' in stats and len(stats['eval_rewards']) > 0:
        eval_episodes = np.arange(0, len(stats['rewards']), args.eval_every)[:len(stats['eval_rewards'])]
        
        axes[1, 1].errorbar(eval_episodes, stats['eval_rewards'],
                          yerr=[s for s in stats.get('eval_reward_ci95s', [0]*len(stats['eval_rewards']))],
                          marker='o', linestyle='-')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Evaluation Reward vs Training Episode')
        axes[1, 1].grid(True)
        
        axes[2, 0].errorbar(eval_episodes, stats['eval_steps'],
                          yerr=[s for s in stats.get('eval_steps_ci95s', [0]*len(stats['eval_steps']))],
                          marker='o', linestyle='-')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Steps')
        axes[2, 0].set_title('Evaluation Steps vs Training Episode')
        axes[2, 0].grid(True)
        
        axes[2, 1].errorbar(eval_episodes, [r*100 for r in stats['eval_completion_rates']],
                          yerr=[s*100 for s in stats.get('eval_completion_rate_ci95s', [0]*len(stats['eval_completion_rates']))],
                          marker='o', linestyle='-')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Completion Rate (%)')
        axes[2, 1].set_title('Evaluation Completion Rate vs Training Episode')
        axes[2, 1].grid(True)
    
    # Plot reward vs wall clock time if available
    if 'wall_time' in stats and len(stats['wall_time']) > 0:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stats['wall_time'], stats['rewards'])
        ax.set_xlabel('Wall Clock Time (s)')
        ax.set_ylabel('Reward')
        ax.set_title('Reward vs Wall Clock Time')
        ax.grid(True)
        
        if save_path:
            fig2.savefig(f"{save_path}_time.png")
    
    # Adjust layout and save if requested
    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}.png")
    
    return fig


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    seed_config = SeedConfig(args.seed)
    
    # Run regression test if requested
    if args.baseline:
        run_regression_test(args)
    
    # Train agent if requested
    if args.train:
        agent, stats = train(args)
        
        # Visualize training statistics
        if args.exp_name:
            visualize_training_stats(stats, save_path=os.path.join(args.log_dir, args.exp_name, 'training_stats'))
        else:
            visualize_training_stats(stats)
    
    # Evaluate agent if requested
    if args.eval and not args.train:
        # Set up environment
        env_params = {
            'task_id': args.task_id,
            'seed': args.seed,
            'max_time_steps': args.max_steps,
            'debug': args.debug
        }
        env = BatteryDisassemblyEnv(env_params)
        
        # Create agent
        agent = create_agent(args, env)
        
        # Load agent if checkpoint exists
        if args.exp_name:
            checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
        else:
            checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.algorithm}_eval")
        
        if os.path.exists(checkpoint_dir):
            checkpoint = find_latest_checkpoint(checkpoint_dir)
            if checkpoint:
                agent.load(os.path.join(checkpoint_dir, checkpoint))
                print(f"Loaded agent from {os.path.join(checkpoint_dir, checkpoint)}")
            else:
                print(f"No checkpoint found in {checkpoint_dir}")
        else:
            print(f"Checkpoint directory {checkpoint_dir} not found")
        
        # Initialize wandb for evaluation if requested
        if args.use_wandb:
            wandb_config = {
                "algorithm": args.algorithm,
                "task_id": args.task_id,
                "eval_episodes": args.n_eval_episodes,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "mode": "evaluation"
            }
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.exp_name or args.algorithm}_eval",
                config=wandb_config
            )
        
        # Evaluate agent
        eval_stats = evaluate(args, agent, env, args.n_eval_episodes, render=args.render)
        
        # Log evaluation results to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "eval/reward": eval_stats['reward'],
                "eval/reward_std": eval_stats['reward_std'],
                "eval/reward_ci95": eval_stats['reward_ci95'],
                "eval/leader_reward": eval_stats['leader_reward'],
                "eval/follower1_reward": eval_stats['follower1_reward'],
                "eval/follower2_reward": eval_stats['follower2_reward'],
                "eval/steps": eval_stats['steps'],
                "eval/steps_std": eval_stats['steps_std'],
                "eval/steps_ci95": eval_stats['steps_ci95'],
                "eval/completion_rate": eval_stats['completion_rate'],
                "eval/completion_rate_ci95": eval_stats['completion_rate_ci95']
            })
            
            # Finish wandb run
            wandb.finish()
        
        # Print evaluation results
        print(f"Evaluation Results ({args.n_eval_episodes} episodes):")
        print(f"Reward: {eval_stats['reward']:.2f} ± {eval_stats['reward_ci95']:.2f}")
        print(f"Leader Reward: {eval_stats['leader_reward']:.2f} ± {1.96 * eval_stats['leader_reward_std'] / np.sqrt(args.n_eval_episodes):.2f}")
        print(f"Follower1 Reward: {eval_stats['follower1_reward']:.2f} ± {1.96 * eval_stats['follower1_reward_std'] / np.sqrt(args.n_eval_episodes):.2f}")
        print(f"Follower2 Reward: {eval_stats['follower2_reward']:.2f} ± {1.96 * eval_stats['follower2_reward_std'] / np.sqrt(args.n_eval_episodes):.2f}")
        print(f"Steps: {eval_stats['steps']:.2f} ± {eval_stats['steps_ci95']:.2f}")
        print(f"Completion Rate: {eval_stats['completion_rate']*100:.1f}% ± {eval_stats['completion_rate_ci95']*100:.1f}%")