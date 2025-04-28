"""
Categorical 51 (C51) Distributional RL implementation for multi-robot coordination.

This module provides the C51 agent implementation that learns value distributions
instead of point estimates for better uncertainty handling in the task.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from agents.base_agent import BaseAgent
from agents.replay_buffer import SequenceReplayBuffer


class C51Network(nn.Module):
    """
    C51 (Categorical 51) Network implementation for distributional Q-learning.
    
    Instead of estimating a single Q-value, it estimates a probability distribution over possible values.
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
        n_atoms (int): Number of atoms in the distribution
        v_min (float): Minimum support value
        v_max (float): Maximum support value
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64, 
                 n_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        """
        Initialize the C51 network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
            n_atoms: Number of atoms in the distribution
            v_min: Minimum support value
            v_max: Maximum support value
        """
        super(C51Network, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_size = hidden_size
        
        # Initialize support for the distribution
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # Output layer for value distribution - outputs logits for softmax
        self.output_layer = nn.Linear(hidden_size, action_dim * n_atoms)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Probability distributions over values for each action [batch_size, action_dim, n_atoms]
        """
        batch_size = state.shape[0]
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Output distribution logits
        logits = self.output_layer(features)
        
        # Reshape to [batch, action_dim, n_atoms]
        logits = logits.view(batch_size, self.action_dim, self.n_atoms)
        
        # Apply softmax to get probability distributions for each action
        probs = torch.softmax(logits, dim=2)
        
        return probs
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convert probability distributions to Q-values.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Q-values [batch_size, action_dim]
        """
        # Get probability distributions
        probs = self.forward(state)
        
        # Expected value: Sum(p_i * z_i) for each action
        q_values = torch.sum(probs * self.support, dim=2)
        
        return q_values


class StackelbergC51Agent(BaseAgent):
    """
    Agent implementation using C51 (Distributional Q-Learning) for Stackelberg games with three robots.
    
    This agent learns probability distributions over Q-values instead of point estimates,
    which allows for better uncertainty handling and more stable learning.
    
    Attributes:
        hidden_size (int): Size of hidden layers
        n_atoms (int): Number of atoms in the distribution
        v_min (float): Minimum support value
        v_max (float): Maximum support value
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate
        epsilon_decay (float): Rate at which epsilon decays over time
        epsilon_min (float): Minimum value for epsilon
        tau (float): Soft update parameter for target network
        update_every (int): How often to update the target network
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, action_dim_follower2: int,
                 hidden_size: int = 64, n_atoms: int = 51, v_min: float = -10, v_max: float = 10, 
                 device: str = 'cpu', learning_rate: float = 1e-4, gamma: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 tau: float = 0.01, update_every: int = 10, seed: int = 42, debug: bool = False):
        """
        Initialize the Stackelberg C51 agent for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            n_atoms: Number of atoms in the distribution
            v_min: Minimum support value
            v_max: Maximum support value
            device: Device to run the model on (cpu or cuda)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays over time
            epsilon_min: Minimum value for epsilon
            tau: Soft update parameter for target network
            update_every: How often to update the target network
            seed: Random seed
            debug: Whether to print debug information
        """
        super().__init__(state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, device, seed)
        
        self.hidden_size = hidden_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_every = update_every
        self.debug = debug
        
        # Initialize support for the distribution
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Initialize leader and follower networks
        self.leader_online = C51Network(
            state_dim, action_dim_leader, hidden_size, n_atoms, v_min, v_max).to(device)
        self.leader_target = C51Network(
            state_dim, action_dim_leader, hidden_size, n_atoms, v_min, v_max).to(device)
        
        self.follower1_online = C51Network(
            state_dim, action_dim_follower1, hidden_size, n_atoms, v_min, v_max).to(device)
        self.follower1_target = C51Network(
            state_dim, action_dim_follower1, hidden_size, n_atoms, v_min, v_max).to(device)
        
        self.follower2_online = C51Network(
            state_dim, action_dim_follower2, hidden_size, n_atoms, v_min, v_max).to(device)
        self.follower2_target = C51Network(
            state_dim, action_dim_follower2, hidden_size, n_atoms, v_min, v_max).to(device)
        
        # Initialize target networks with same weights as online networks
        self.leader_target.load_state_dict(self.leader_online.state_dict())
        self.follower1_target.load_state_dict(self.follower1_online.state_dict())
        self.follower2_target.load_state_dict(self.follower2_online.state_dict())
        
        # Initialize optimizers
        self.leader_optimizer = optim.Adam(self.leader_online.parameters(), lr=learning_rate)
        self.follower1_optimizer = optim.Adam(self.follower1_online.parameters(), lr=learning_rate)
        self.follower2_optimizer = optim.Adam(self.follower2_online.parameters(), lr=learning_rate)
        
        # Initialize training step counter
        self.t_step = 0
    
    def compute_stackelberg_equilibrium(self, state: np.ndarray, 
                                       action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Compute Stackelberg equilibrium using the current networks.
        In this hierarchy: Leader -> (Follower1, Follower2)
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Process action masks if provided
        if action_masks is not None:
            leader_mask, follower1_mask, follower2_mask = self.process_action_mask(action_masks)
        else:
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Get Q-values for all possible actions
        with torch.no_grad():
            leader_q_values = self.leader_online.get_q_values(state_tensor)
            follower1_q_values = self.follower1_online.get_q_values(state_tensor)
            follower2_q_values = self.follower2_online.get_q_values(state_tensor)
        
        # Apply action masks
        leader_q_values = self.apply_action_mask(leader_q_values, leader_mask)
        follower1_q_values = self.apply_action_mask(follower1_q_values, follower1_mask)
        follower2_q_values = self.apply_action_mask(follower2_q_values, follower2_mask)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()[0]
        follower1_q = follower1_q_values.detach().cpu().numpy()[0]
        follower2_q = follower2_q_values.detach().cpu().numpy()[0]
        
        # For each potential leader action, compute the Nash equilibrium between the followers
        best_leader_value = float('-inf')
        leader_se_action = 0
        follower1_se_action = 0
        follower2_se_action = 0
        
        for a_l in range(self.action_dim_leader):
            if not leader_mask[a_l].item():
                continue  # Skip invalid leader actions
                
            # Initialize with a suboptimal solution
            f1_action, f2_action = 0, 0
            
            # Simple iterative best response for the followers' subgame
            for _ in range(10):  # Few iterations usually converge
                # Follower 1's best response to current follower 2's action
                valid_f1_actions = np.where(follower1_mask.cpu().numpy())[0]
                f1_best_response = valid_f1_actions[np.argmax(follower1_q[valid_f1_actions])] if len(valid_f1_actions) > 0 else 0
                
                # Follower 2's best response to updated follower 1's action
                valid_f2_actions = np.where(follower2_mask.cpu().numpy())[0]
                f2_best_response = valid_f2_actions[np.argmax(follower2_q[valid_f2_actions])] if len(valid_f2_actions) > 0 else 0
                
                # Update actions
                if f1_action == f1_best_response and f2_action == f2_best_response:
                    break  # Equilibrium reached
                    
                f1_action, f2_action = f1_best_response, f2_best_response
            
            # Evaluate leader's utility with this followers' equilibrium
            leader_value = leader_q[a_l]
            
            if leader_value > best_leader_value:
                best_leader_value = leader_value
                leader_se_action = a_l
                follower1_se_action = f1_action
                follower2_se_action = f2_action
        
        # Convert from index to actual action (-1 to n-2, where n is action_dim)
        return leader_se_action - 1, follower1_se_action - 1, follower2_se_action - 1
    
    def act(self, state: np.ndarray, action_masks: Optional[Dict[str, np.ndarray]] = None, 
            epsilon: Optional[float] = None) -> Tuple[int, int, int]:
        """
        Select actions according to epsilon-greedy policy.
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            epsilon: Exploration rate (uses default if None)
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # With probability epsilon, select random actions
        if np.random.random() < epsilon:
            if action_masks is not None:
                # Select random valid actions using masks
                valid_leader_actions = np.where(action_masks['leader'])[0]
                valid_follower1_actions = np.where(action_masks['follower1'])[0]
                valid_follower2_actions = np.where(action_masks['follower2'])[0]
                
                if len(valid_leader_actions) > 0:
                    leader_action_idx = np.random.choice(valid_leader_actions)
                else:
                    leader_action_idx = 0  # Default to "do nothing"
                    
                if len(valid_follower1_actions) > 0:
                    follower1_action_idx = np.random.choice(valid_follower1_actions)
                else:
                    follower1_action_idx = 0  # Default to "do nothing"
                    
                if len(valid_follower2_actions) > 0:
                    follower2_action_idx = np.random.choice(valid_follower2_actions)
                else:
                    follower2_action_idx = 0  # Default to "do nothing"
                
                # Convert to actual actions (-1 if action_idx is 0, otherwise action_idx - 1)
                leader_action = leader_action_idx - 1
                follower1_action = follower1_action_idx - 1
                follower2_action = follower2_action_idx - 1
            else:
                # No masks provided, select from full action space
                leader_action = np.random.randint(-1, self.action_dim_leader - 1)
                follower1_action = np.random.randint(-1, self.action_dim_follower1 - 1)
                follower2_action = np.random.randint(-1, self.action_dim_follower2 - 1)
            
            return leader_action, follower1_action, follower2_action
        
        # Otherwise, compute and return Stackelberg equilibrium actions
        return self.compute_stackelberg_equilibrium(state, action_masks)
    
    def project_distribution(self, rewards: torch.Tensor, next_probs: torch.Tensor, 
                             dones: torch.Tensor) -> torch.Tensor:
        """
        Project the distribution for Categorical DQN algorithm.
        
        Args:
            rewards: Batch of rewards [batch_size]
            next_probs: Probability distributions from target network [batch_size, n_atoms]
            dones: Batch of done flags [batch_size]
            
        Returns:
            Projected distribution [batch_size, n_atoms]
        """
        batch_size = rewards.shape[0]
        projected_dist = torch.zeros(batch_size, self.n_atoms, device=self.device)
        
        # For terminal states, just return the immediate reward distribution
        for i in range(batch_size):
            if dones[i]:
                # For terminal states, just the immediate reward matters
                tz = torch.clamp(rewards[i], self.v_min, self.v_max)
                bj = ((tz - self.v_min) / self.delta_z).floor().long()
                l = (tz - self.v_min - bj * self.delta_z) / self.delta_z
                u = 1.0 - l
                
                # Zeroing the projected distribution
                projected_dist[i].fill_(0.0)
                projected_dist[i, bj] = u
                if bj < self.n_atoms - 1:
                    projected_dist[i, bj + 1] = l
            else:
                # For non-terminal states, we need to project the distribution
                for j in range(self.n_atoms):
                    # Apply Bellman update
                    tz = torch.clamp(rewards[i] + self.gamma * self.support[j], self.v_min, self.v_max)
                    bj = ((tz - self.v_min) / self.delta_z).floor().long()
                    l = (tz - self.v_min - bj * self.delta_z) / self.delta_z
                    u = 1.0 - l
                    
                    # Distribute probability mass
                    projected_dist[i, bj] += next_probs[i, j] * u
                    if bj < self.n_atoms - 1:
                        projected_dist[i, bj + 1] += next_probs[i, j] * l
        
        return projected_dist
    
    def update(self, experiences: List[Tuple]) -> Tuple[float, float, float]:
        """
        Update the networks using a batch of experiences.
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            Losses for leader, follower1, and follower2
        """
        # Process experiences
        states = []
        leader_actions = []
        follower1_actions = []
        follower2_actions = []
        leader_rewards = []
        follower1_rewards = []
        follower2_rewards = []
        next_states = []
        dones = []
        
        for exp in experiences:
            s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, s_next, done = exp
            
            # Convert actions to indices (add 1 to handle -1 actions)
            a_l_idx = a_l + 1
            a_f1_idx = a_f1 + 1
            a_f2_idx = a_f2 + 1
            
            states.append(s)
            leader_actions.append(a_l_idx)
            follower1_actions.append(a_f1_idx)
            follower2_actions.append(a_f2_idx)
            leader_rewards.append(r_l)
            follower1_rewards.append(r_f1)
            follower2_rewards.append(r_f2)
            next_states.append(s_next)
            dones.append(done)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        leader_actions = torch.tensor(leader_actions, dtype=torch.long).to(self.device)
        follower1_actions = torch.tensor(follower1_actions, dtype=torch.long).to(self.device)
        follower2_actions = torch.tensor(follower2_actions, dtype=torch.long).to(self.device)
        leader_rewards = torch.tensor(leader_rewards, dtype=torch.float).to(self.device)
        follower1_rewards = torch.tensor(follower1_rewards, dtype=torch.float).to(self.device)
        follower2_rewards = torch.tensor(follower2_rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        batch_size = states.shape[0]
        
        # Compute current probability distributions
        leader_probs = self.leader_online(states)
        follower1_probs = self.follower1_online(states)
        follower2_probs = self.follower2_online(states)
        
        # Extract distributions for the actions taken
        leader_probs_taken = leader_probs[torch.arange(batch_size), leader_actions]
        follower1_probs_taken = follower1_probs[torch.arange(batch_size), follower1_actions]
        follower2_probs_taken = follower2_probs[torch.arange(batch_size), follower2_actions]
        
        # Compute next state distributions using target networks
        with torch.no_grad():
            # Get Q-values from online networks
            next_leader_q_online = self.leader_online.get_q_values(next_states)
            next_follower1_q_online = self.follower1_online.get_q_values(next_states)
            next_follower2_q_online = self.follower2_online.get_q_values(next_states)
            
            # Select best actions using online network (double DQN)
            next_leader_actions = next_leader_q_online.argmax(dim=1)
            next_follower1_actions = next_follower1_q_online.argmax(dim=1)
            next_follower2_actions = next_follower2_q_online.argmax(dim=1)
            
            # Get distributions from target networks
            next_leader_probs = self.leader_target(next_states)
            next_follower1_probs = self.follower1_target(next_states)
            next_follower2_probs = self.follower2_target(next_states)
            
            # Extract target distributions for selected actions
            next_leader_dist = next_leader_probs[torch.arange(batch_size), next_leader_actions]
            next_follower1_dist = next_follower1_probs[torch.arange(batch_size), next_follower1_actions]
            next_follower2_dist = next_follower2_probs[torch.arange(batch_size), next_follower2_actions]
            
            # Project distributions for each agent
            projected_leader_dist = self.project_distribution(leader_rewards, next_leader_dist, dones)
            projected_follower1_dist = self.project_distribution(follower1_rewards, next_follower1_dist, dones)
            projected_follower2_dist = self.project_distribution(follower2_rewards, next_follower2_dist, dones)
        
        # Compute cross-entropy loss for each agent
        leader_loss = -(projected_leader_dist * torch.log(leader_probs_taken + 1e-10)).sum(dim=1).mean()
        follower1_loss = -(projected_follower1_dist * torch.log(follower1_probs_taken + 1e-10)).sum(dim=1).mean()
        follower2_loss = -(projected_follower2_dist * torch.log(follower2_probs_taken + 1e-10)).sum(dim=1).mean()
        
        # Update online networks
        # Leader
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1.0)
        self.leader_optimizer.step()
        
        # Follower 1
        self.follower1_optimizer.zero_grad()
        follower1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 1.0)
        self.follower1_optimizer.step()
        
        # Follower 2
        self.follower2_optimizer.zero_grad()
        follower2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 1.0)
        self.follower2_optimizer.step()
        
        # Soft update target networks
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            self.soft_update(self.leader_online, self.leader_target)
            self.soft_update(self.follower1_online, self.follower1_target)
            self.soft_update(self.follower2_online, self.follower2_target)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return leader_loss.item(), follower1_loss.item(), follower2_loss.item()
    
    def soft_update(self, online_model: nn.Module, target_model: nn.Module) -> None:
        """
        Soft update of target network parameters.
        θ_target = τ*θ_online + (1 - τ)*θ_target
        
        Args:
            online_model: Online network
            target_model: Target network
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.leader_online.state_dict(), f"{path}/leader_online.pt")
        torch.save(self.leader_target.state_dict(), f"{path}/leader_target.pt")
        torch.save(self.follower1_online.state_dict(), f"{path}/follower1_online.pt")
        torch.save(self.follower1_target.state_dict(), f"{path}/follower1_target.pt")
        torch.save(self.follower2_online.state_dict(), f"{path}/follower2_online.pt")
        torch.save(self.follower2_target.state_dict(), f"{path}/follower2_target.pt")
        
        params = {
            "epsilon": self.epsilon,
            "t_step": self.t_step,
            "hidden_size": self.hidden_size,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max
        }
        
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(params, f)
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Directory to load from
        """
        self.leader_online.load_state_dict(torch.load(f"{path}/leader_online.pt", map_location=self.device))
        self.leader_target.load_state_dict(torch.load(f"{path}/leader_target.pt", map_location=self.device))
        self.follower1_online.load_state_dict(torch.load(f"{path}/follower1_online.pt", map_location=self.device))
        self.follower1_target.load_state_dict(torch.load(f"{path}/follower1_target.pt", map_location=self.device))
        self.follower2_online.load_state_dict(torch.load(f"{path}/follower2_online.pt", map_location=self.device))
        self.follower2_target.load_state_dict(torch.load(f"{path}/follower2_target.pt", map_location=self.device))
        
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
            self.epsilon = params["epsilon"]
            self.t_step = params["t_step"]
            # Check if the loaded model has the same configuration
            if "hidden_size" in params and params["hidden_size"] != self.hidden_size:
                print(f"Warning: Loaded model has hidden size {params['hidden_size']}, but current model has {self.hidden_size}")
            if "n_atoms" in params and params["n_atoms"] != self.n_atoms:
                print(f"Warning: Loaded model has n_atoms={params['n_atoms']}, but current model has n_atoms={self.n_atoms}")
            if "v_min" in params and params["v_min"] != self.v_min:
                print(f"Warning: Loaded model has v_min={params['v_min']}, but current model has v_min={self.v_min}")
            if "v_max" in params and params["v_max"] != self.v_max:
                print(f"Warning: Loaded model has v_max={params['v_max']}, but current model has v_max={self.v_max}")