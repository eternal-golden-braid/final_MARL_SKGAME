#!/usr/bin/env python3
"""
Main run script for multi-robot battery disassembly coordination.

This script provides a unified interface for training, evaluating, and testing
the different reinforcement learning algorithms for the battery disassembly task.
"""
import os
import sys
import argparse
import subprocess
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any

def setup_environment():
    """
    Set up the environment by checking dependencies and creating directories.
    """
    # Check for required libraries
    try:
        import numpy
        import torch
        import matplotlib
        import tensorboardX
        print("All required libraries are installed.")
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install all required libraries:")
        print("pip install torch numpy matplotlib tensorboardX tqdm wandb")
        sys.exit(1)
    
    # Create required directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Set PyTorch seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Environment setup complete.")


def run_training(args):
    """
    Run training for the specified algorithm.
    
    Args:
        args: Command line arguments
    """
    print(f"Starting training for {args.algorithm}...")
    
    # Construct command line arguments for train.py
    cmd = [
        "python", "train.py",
        "--algorithm", args.algorithm,
        "--train",
        "--n_episodes", str(args.n_episodes),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden_size", str(args.hidden_size),
        "--gamma", str(args.gamma),
        "--epsilon", str(args.epsilon),
        "--epsilon_decay", str(args.epsilon_decay),
        "--epsilon_min", str(args.epsilon_min),
        "--task_id", str(args.task_id),
        "--max_steps", str(args.max_steps),
        "--seed", str(args.seed)
    ]
    
    # Add optional arguments
    if args.gpu:
        cmd.append("--gpu")
    if args.render:
        cmd.append("--render")
    if args.use_qmix:
        cmd.append("--use_qmix")
    if args.exp_name:
        cmd.extend(["--exp_name", args.exp_name])
    else:
        # Create a default experiment name if not provided
        exp_name = f"{args.algorithm}_{args.hidden_size}_{args.lr}_{int(time.time())}"
        cmd.extend(["--exp_name", exp_name])
        args.exp_name = exp_name
    
    # Add algorithm-specific arguments
    if args.algorithm == "drqn":
        cmd.extend(["--lstm_layers", str(args.lstm_layers)])
    elif args.algorithm == "c51":
        cmd.extend([
            "--n_atoms", str(args.n_atoms),
            "--v_min", str(args.v_min),
            "--v_max", str(args.v_max)
        ])
    
    # Add wandb arguments if specified
    if args.use_wandb:
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
    
    # Run the command
    subprocess.run(cmd)
    
    print(f"Training completed for {args.algorithm}.")


def run_evaluation(args):
    """
    Run evaluation for the specified algorithm.
    
    Args:
        args: Command line arguments
    """
    print(f"Starting evaluation for {args.algorithm}...")
    
    # Construct command line arguments for train.py with --eval flag
    cmd = [
        "python", "train.py",
        "--algorithm", args.algorithm,
        "--eval",
        "--n_eval_episodes", str(args.n_eval_episodes),
        "--task_id", str(args.task_id),
        "--max_steps", str(args.max_steps),
        "--seed", str(args.seed)
    ]
    
    # Add experiment name
    if args.exp_name:
        cmd.extend(["--exp_name", args.exp_name])
    
    # Add optional arguments
    if args.gpu:
        cmd.append("--gpu")
    if args.render:
        cmd.append("--render")
    
    # Add wandb arguments if specified
    if args.use_wandb:
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
    
    # Run the command
    subprocess.run(cmd)
    
    print(f"Evaluation completed for {args.algorithm}.")


def run_regression_test(args):
    """
    Run regression test for the DRQN algorithm.
    
    Args:
        args: Command line arguments
    """
    print("Starting DRQN regression test...")
    
    # Construct command line arguments for drqn_regression_test.py
    cmd = [
        "python", "tests/drqn_regression_test.py",
        "--n_episodes", str(args.n_episodes),
        "--seed", str(args.seed)
    ]
    
    # Add optional arguments
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.baseline_reward is not None:
        cmd.extend(["--baseline_reward", str(args.baseline_reward)])
    if args.render:
        cmd.append("--render")
    if args.save_baseline:
        cmd.append("--save_baseline")
    
    # Run the command
    subprocess.run(cmd)
    
    print("DRQN regression test completed.")


def run_comparative_test(args):
    """
    Run comparative test for multiple algorithms.
    
    Args:
        args: Command line arguments
    """
    print("Starting comparative test...")
    
    # Construct command line arguments for main_test_script.py
    cmd = [
        "python", "tests/main_test_script.py",
        "--n_episodes", str(args.n_episodes),
        "--n_runs", str(args.n_runs),
        "--seed", str(args.seed),
        "--task_id", str(args.task_id)
    ]
    
    # Add algorithms
    if args.algorithms:
        cmd.extend(["--algorithms"] + args.algorithms)
    
    # Add optional arguments
    if args.render:
        cmd.append("--render")
    if args.save_plot:
        cmd.extend(["--save_plot", args.save_plot])
    if args.save_results:
        cmd.extend(["--save_results", args.save_results])
    
    # Run the command
    subprocess.run(cmd)
    
    print("Comparative test completed.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-robot disassembly coordination')
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up the environment')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--algorithm', type=str, default='drqn', choices=['drqn', 'ddqn', 'c51', 'sac'],
                           help='Algorithm to use: drqn, ddqn, c51, sac')
    train_parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for networks')
    train_parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=0.1, help='Initial epsilon for exploration')
    train_parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    train_parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    train_parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    train_parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    train_parser.add_argument('--render', action='store_true', help='Render environment')
    train_parser.add_argument('--use_qmix', action='store_true', help='Use QMIX for value factorization')
    train_parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    train_parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers (DRQN only)')
    train_parser.add_argument('--n_atoms', type=int, default=51, help='Number of atoms (C51 only)')
    train_parser.add_argument('--v_min', type=float, default=-10, help='Minimum value (C51 only)')
    train_parser.add_argument('--v_max', type=float, default=10, help='Maximum value (C51 only)')
    train_parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    train_parser.add_argument('--wandb_project', type=str, default='multi-robot-disassembly', help='Weights & Biases project name')
    train_parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate an agent')
    eval_parser.add_argument('--algorithm', type=str, default='drqn', choices=['drqn', 'ddqn', 'c51', 'sac'],
                          help='Algorithm to use: drqn, ddqn, c51, sac')
    eval_parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    eval_parser.add_argument('--n_eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    eval_parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    eval_parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    eval_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    eval_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    eval_parser.add_argument('--render', action='store_true', help='Render environment')
    eval_parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    eval_parser.add_argument('--wandb_project', type=str, default='multi-robot-disassembly', help='Weights & Biases project name')
    eval_parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity')
    
    # Regression test command
    regression_parser = subparsers.add_parser('regression', help='Run regression test for DRQN')
    regression_parser.add_argument('--n_episodes', type=int, default=20, help='Number of episodes')
    regression_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    regression_parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    regression_parser.add_argument('--baseline_reward', type=float, default=-5.0, help='Minimum average reward to pass the test')
    regression_parser.add_argument('--render', action='store_true', help='Render environment')
    regression_parser.add_argument('--save_baseline', action='store_true', help='Save baseline performance')
    
    # Comparative test command
    comparative_parser = subparsers.add_parser('comparative', help='Run comparative test for multiple algorithms')
    comparative_parser.add_argument('--algorithms', nargs='+', default=['drqn', 'ddqn', 'c51'], help='Algorithms to test')
    comparative_parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes')
    comparative_parser.add_argument('--n_runs', type=int, default=5, help='Number of independent runs')
    comparative_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    comparative_parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    comparative_parser.add_argument('--render', action='store_true', help='Render environment')
    comparative_parser.add_argument('--save_plot', type=str, default='results/comparative_results.png', help='Path to save the results plot')
    comparative_parser.add_argument('--save_results', type=str, default='results/comparative_results.csv', help='Path to save the results as CSV')
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    if args.command == 'setup':
        setup_environment()
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'eval':
        run_evaluation(args)
    elif args.command == 'regression':
        run_regression_test(args)
    elif args.command == 'comparative':
        run_comparative_test(args)
    else:
        print("Please specify a command: setup, train, eval, regression, or comparative")
        print("Run with --help for more information")


if __name__ == "__main__":
    main()