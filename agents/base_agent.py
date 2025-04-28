"""
Base agent classes for multi-robot reinforcement learning.

This module provides the base agent classes that all specific
agent implementations will inherit from.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any


class BaseAgent:
    """
    Base class for all agents.
    
    This class provides common functionality for all agents.
    
    Attributes:
        state_dim (int): Dimension of the state space
        action_dim_leader (int): Dimension of the leader's action space
        action_dim_follower1 (int): Dimension of follower1's action space
        action_dim_follower2 (int): Dimension of follower2's action space
        device (str): Device to run the model on (cpu or cuda)
        seed (int): Random seed
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, 
                 action_dim_follower2: int, device: str = 'cpu', seed: int = 42):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            device: Device to run the model on (cpu or cuda)
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.device = device
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize debug flag
        self.debug = False
    
    def update(self, *args, **kwargs):
        """
        Update the agent's networks.
        
        Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def act(self, *args, **kwargs):
        """
        Select actions for all robots.
        
        Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Directory to save to
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Directory to load from
        """
        raise NotImplementedError
    
    def apply_action_mask(self, q_values: torch.Tensor, action_mask: torch.Tensor, 
                        min_value: float = -1e8) -> torch.Tensor:
        """
        Apply an action mask to q-values.
        
        Args:
            q_values: Q-values tensor
            action_mask: Boolean mask tensor (True for valid actions)
            min_value: Value to use for masked (invalid) actions
            
        Returns:
            Masked Q-values tensor with invalid actions set to min_value
        """
        # Convert boolean mask to float
        mask = action_mask.float()
        
        # Apply mask: set invalid actions to min_value
        return q_values * mask + (1 - mask) * min_value
        
    def process_action_mask(self, state_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process action masks from environment state.
        
        Args:
            state_dict: Dictionary containing state information including action masks
            
        Returns:
            Tuple of action masks for leader, follower1, and follower2 as tensors
        """
        if 'action_masks' not in state_dict:
            # If no mask provided, all actions are valid
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool)
        else:
            # Convert numpy masks to torch tensors
            leader_mask = torch.tensor(state_dict['action_masks']['leader'], dtype=torch.bool)
            follower1_mask = torch.tensor(state_dict['action_masks']['follower1'], dtype=torch.bool)
            follower2_mask = torch.tensor(state_dict['action_masks']['follower2'], dtype=torch.bool)
        
        return leader_mask.to(self.device), follower1_mask.to(self.device), follower2_mask.to(self.device)


class QMIXNetwork(nn.Module):
    """
    QMIX network for value factorization.
    
    This network combines individual agent Q-values into a joint Q-value
    while maintaining monotonicity to enable efficient training.
    
    Attributes:
        state_dim (int): Dimension of the state space
        hidden_size (int): Size of hidden layers
    """
    def __init__(self, state_dim: int, hidden_size: int = 64):
        """
        Initialize the QMIX network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_size: Size of hidden layers
        """
        super(QMIXNetwork, self).__init__()
        
        # Hypernetwork that generates weights for the first layer of mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 3 agents: leader, follower1, follower2
        )
        
        # Hypernetwork that generates weights for the second layer of mixing network
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single weight for the final layer
        )
        
        # Hypernetwork that generates bias for the first layer of mixing network
        self.hyper_b1 = nn.Linear(state_dim, 1)
        
        # Hypernetwork that generates bias for the second layer of mixing network
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, agent_q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QMIX network.
        
        Args:
            agent_q_values: Q-values from individual agents [batch_size, 3]
            states: Global states [batch_size, state_dim]
            
        Returns:
            Joint Q-values [batch_size, 1]
        """
        # Get batch size
        batch_size = agent_q_values.shape[0]
        
        # Generate weights and biases from hypernetworks
        w1 = self.hyper_w1(states).view(batch_size, 1, 3)   # [batch_size, 1, 3]
        b1 = self.hyper_b1(states).view(batch_size, 1, 1)   # [batch_size, 1, 1]
        
        # Ensure weights are positive for monotonicity
        w1 = torch.abs(w1)
        
        # First layer of mixing network
        hidden = torch.bmm(w1, agent_q_values.unsqueeze(2)).view(batch_size, 1) + b1.view(batch_size, 1)
        hidden = torch.relu(hidden)
        
        # Generate weights and biases for the second layer
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, 1, 1)   # [batch_size, 1, 1]
        b2 = self.hyper_b2(states).view(batch_size, 1)                # [batch_size, 1]
        
        # Second layer of mixing network
        q_total = torch.bmm(w2, hidden.unsqueeze(2)).view(batch_size, 1) + b2
        
        return q_total
        
    def get_joint_q_value(self, agent_q_values: List[torch.Tensor], states: torch.Tensor) -> torch.Tensor:
        """
        Calculate joint Q-value from individual agent Q-values.
        
        Args:
            agent_q_values: List of Q-values from individual agents [leader, follower1, follower2]
            states: Global states [batch_size, state_dim]
            
        Returns:
            Joint Q-values
        """
        # Stack agent Q-values along dim 1 (agent dimension)
        # Handle the case where q_values might have different shapes
        batch_size = agent_q_values[0].shape[0]
        
        # Reshape if needed (for sequence data)
        reshaped_q_values = []
        for q in agent_q_values:
            if len(q.shape) > 2:  # If q has sequence dimension
                reshaped_q_values.append(q.reshape(batch_size, -1))
            else:
                reshaped_q_values.append(q)
        
        # Stack along agent dimension
        agent_qs = torch.stack(reshaped_q_values, dim=1)  # [batch_size, 3]
        
        # Reshape states if needed
        if len(states.shape) > 2:  # If states has sequence dimension
            states = states.reshape(batch_size, -1)
        
        # Forward pass through the mixing network
        return self.forward(agent_qs, states)