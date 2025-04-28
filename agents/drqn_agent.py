"""
Deep Recurrent Q-Network implementation for multi-robot coordination.

This module provides the DRQN agent implementation that uses recurrent
neural networks to handle temporal dependencies in the environment.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from agents.base_agent import BaseAgent, QMIXNetwork
from agents.replay_buffer import SequenceReplayBuffer


class RecurrentQNetwork(nn.Module):
    """
    Deep Recurrent Q-Network implementation using LSTM.
    This handles temporal dependencies through recurrent layers.
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
        lstm_layers (int): Number of LSTM layers
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64, lstm_layers: int = 1):
        """
        Initialize the recurrent Q-network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
            lstm_layers: Number of LSTM layers
        """
        super(RecurrentQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output layer for Q-values
        self.output_layer = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            state: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Initial hidden state for LSTM
        
        Returns:
            Q-values and final hidden state
        """
        batch_size, seq_len, _ = state.shape
        
        # Extract features
        features = self.feature_extractor(state.reshape(-1, self.input_dim))
        features = features.view(batch_size, seq_len, self.hidden_size)
        
        # Pass through LSTM
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features)
        else:
            lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Generate Q-values
        q_values = self.output_layer(lstm_out)
        
        return q_values, hidden_state
    
    def get_q_values(self, state: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get Q-values for a single state.
        
        Args:
            state: Single state tensor [state_dim] or [batch_size, state_dim]
            hidden_state: Hidden state for LSTM
        
        Returns:
            Q-values and updated hidden state
        """
        # Add batch and sequence dimensions if not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        elif len(state.shape) == 2:
            state = state.unsqueeze(1)  # [batch_size, 1, state_dim]
        
        # Forward pass
        q_values, new_hidden_state = self.forward(state, hidden_state)
        
        # Return last timestep's Q-values
        return q_values[:, -1, :], new_hidden_state


class StackelbergDRQNAgent(BaseAgent):
    """
    Agent implementation using Deep Recurrent Q-Networks for Stackelberg games with three robots.
    
    This agent uses recurrent networks to maintain temporal dependencies in the environment.
    It implements a Stackelberg hierarchy where the leader acts first, followed by the followers.
    
    Attributes:
        hidden_size (int): Size of hidden layers
        sequence_length (int): Length of sequences for training
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate
        epsilon_decay (float): Rate at which epsilon decays over time
        epsilon_min (float): Minimum value for epsilon
        tau (float): Soft update parameter for target network
        update_every (int): How often to update the target network
        use_qmix (bool): Whether to use QMIX for mixing individual agent Q-values
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, action_dim_follower2: int,
                 hidden_size: int = 64, sequence_length: int = 8, device: str = 'cpu', learning_rate: float = 1e-4,
                 gamma: float = 0.9, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 tau: float = 0.01, update_every: int = 10, use_qmix: bool = False, lstm_layers: int = 1, 
                 seed: int = 42, debug: bool = False):
        """
        Initialize the Stackelberg DRQN agent for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            sequence_length: Length of sequences for training
            device: Device to run the model on (cpu or cuda)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays over time
            epsilon_min: Minimum value for epsilon
            tau: Soft update parameter for target network
            update_every: How often to update the target network
            use_qmix: Whether to use QMIX for mixing individual agent Q-values
            lstm_layers: Number of LSTM layers
            seed: Random seed
            debug: Whether to print debug information
        """
        super().__init__(state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, device, seed)
        
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_every = update_every
        self.use_qmix = use_qmix
        self.debug = debug
        
        # Initialize leader and follower networks
        self.leader_online = RecurrentQNetwork(
            state_dim, action_dim_leader, hidden_size, lstm_layers).to(device)
        self.leader_target = RecurrentQNetwork(
            state_dim, action_dim_leader, hidden_size, lstm_layers).to(device)
        
        self.follower1_online = RecurrentQNetwork(
            state_dim, action_dim_follower1, hidden_size, lstm_layers).to(device)
        self.follower1_target = RecurrentQNetwork(
            state_dim, action_dim_follower1, hidden_size, lstm_layers).to(device)
        
        self.follower2_online = RecurrentQNetwork(
            state_dim, action_dim_follower2, hidden_size, lstm_layers).to(device)
        self.follower2_target = RecurrentQNetwork(
            state_dim, action_dim_follower2, hidden_size, lstm_layers).to(device)
        
        # Initialize target networks with same weights as online networks
        self.leader_target.load_state_dict(self.leader_online.state_dict())
        self.follower1_target.load_state_dict(self.follower1_online.state_dict())
        self.follower2_target.load_state_dict(self.follower2_online.state_dict())
        
        # Initialize optimizers
        self.leader_optimizer = optim.Adam(self.leader_online.parameters(), lr=learning_rate)
        self.follower1_optimizer = optim.Adam(self.follower1_online.parameters(), lr=learning_rate)
        self.follower2_optimizer = optim.Adam(self.follower2_online.parameters(), lr=learning_rate)
        
        # Initialize QMIX network if enabled
        if use_qmix:
            self.qmix_online = QMIXNetwork(state_dim, hidden_size).to(device)
            self.qmix_target = QMIXNetwork(state_dim, hidden_size).to(device)
            self.qmix_target.load_state_dict(self.qmix_online.state_dict())
            self.qmix_optimizer = optim.Adam(self.qmix_online.parameters(), lr=learning_rate)
        
        # Initialize hidden states
        self.reset_hidden_states()
        
        # Initialize training step counter
        self.t_step = 0
    
    def reset_hidden_states(self) -> None:
        """Reset the hidden states for all recurrent networks."""
        self.leader_hidden = None
        self.follower1_hidden = None
        self.follower2_hidden = None
    
    def compute_stackelberg_equilibrium(self, state: np.ndarray, 
                                       action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Compute Stackelberg equilibrium using the current Q-networks.
        In this hierarchy: Leader -> (Follower1, Follower2)
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Process action masks if provided
        if action_masks is not None:
            leader_mask, follower1_mask, follower2_mask = self.process_action_mask(action_masks)
        else:
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Get Q-values for all possible action combinations
        leader_q_values, self.leader_hidden = self.leader_online.get_q_values(
            state_tensor, self.leader_hidden)
        follower1_q_values, self.follower1_hidden = self.follower1_online.get_q_values(
            state_tensor, self.follower1_hidden)
        follower2_q_values, self.follower2_hidden = self.follower2_online.get_q_values(
            state_tensor, self.follower2_hidden)
        
        # Apply action masks
        leader_q_values = self.apply_action_mask(leader_q_values, leader_mask)
        follower1_q_values = self.apply_action_mask(follower1_q_values, follower1_mask)
        follower2_q_values = self.apply_action_mask(follower2_q_values, follower2_mask)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()
        follower1_q = follower1_q_values.detach().cpu().numpy()
        follower2_q = follower2_q_values.detach().cpu().numpy()
        
        # Note: Original implementation had a structure like:
        # For each potential leader action, compute the Nash equilibrium between the followers
        # with iterative best response. I'm keeping this structure unchanged to maintain
        # the DRQN's behavior.
        
        # For each potential leader action, compute the Nash equilibrium between the followers
        best_leader_value = float('-inf')
        leader_se_action = 0
        follower1_se_action = 0
        follower2_se_action = 0
        
        for a_l in range(self.action_dim_leader):
            if not leader_mask[a_l].item():
                continue  # Skip invalid leader actions
                
            # Followers make independent decisions (not full Nash equilibrium)
            f1_action = np.argmax(follower1_q)
            f2_action = np.argmax(follower2_q)
            
            # Evaluate leader's utility with these follower actions
            leader_value = leader_q[a_l]
            
            if leader_value > best_leader_value:
                best_leader_value = leader_value
                leader_se_action = a_l
                follower1_se_action = f1_action
                follower2_se_action = f2_action
        
        # Convert from index to actual action (-1 to n-2, where n is action_dim)
        # Note: This assumes the first action (index 0) corresponds to "do nothing" (-1)
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
    
    def update(self, experiences: List[List[Tuple]]) -> Tuple[float, float, float]:
        """
        Update the Q-networks using a batch of experiences.
        
        Args:
            experiences: List of sequences of experience tuples
            
        Returns:
            Losses for leader, follower1, and follower2
        """
        # Convert experiences to tensors
        states = []
        leader_actions = []
        follower1_actions = []
        follower2_actions = []
        leader_rewards = []
        follower1_rewards = []
        follower2_rewards = []
        next_states = []
        dones = []
        
        # Process each sequence of experiences
        for sequence in experiences:
            seq_states = []
            seq_leader_actions = []
            seq_follower1_actions = []
            seq_follower2_actions = []
            seq_leader_rewards = []
            seq_follower1_rewards = []
            seq_follower2_rewards = []
            seq_next_states = []
            seq_dones = []
            
            for exp in sequence:
                s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, s_next, done = exp
                
                # Convert actions to indices (add 1 to handle -1 actions)
                a_l_idx = a_l + 1
                a_f1_idx = a_f1 + 1
                a_f2_idx = a_f2 + 1
                
                seq_states.append(s)
                seq_leader_actions.append(a_l_idx)
                seq_follower1_actions.append(a_f1_idx)
                seq_follower2_actions.append(a_f2_idx)
                seq_leader_rewards.append(r_l)
                seq_follower1_rewards.append(r_f1)
                seq_follower2_rewards.append(r_f2)
                seq_next_states.append(s_next)
                seq_dones.append(done)
            
            states.append(seq_states)
            leader_actions.append(seq_leader_actions)
            follower1_actions.append(seq_follower1_actions)
            follower2_actions.append(seq_follower2_actions)
            leader_rewards.append(seq_leader_rewards)
            follower1_rewards.append(seq_follower1_rewards)
            follower2_rewards.append(seq_follower2_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
        
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
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Compute Q-values for current states
        leader_q_values, _ = self.leader_online(states)
        follower1_q_values, _ = self.follower1_online(states)
        follower2_q_values, _ = self.follower2_online(states)
        
        # Gather Q-values for the actions taken
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        leader_q = leader_q_values[batch_indices, seq_indices, leader_actions]
        follower1_q = follower1_q_values[batch_indices, seq_indices, follower1_actions]
        follower2_q = follower2_q_values[batch_indices, seq_indices, follower2_actions]
        
        # Compute next state targets (Q-learning)
        with torch.no_grad():
            # Get target Q-values
            next_leader_q_values, _ = self.leader_target(next_states)
            next_follower1_q_values, _ = self.follower1_target(next_states)
            next_follower2_q_values, _ = self.follower2_target(next_states)
            
            # Get max Q-values for next states
            next_leader_q = next_leader_q_values.max(dim=2)[0]
            next_follower1_q = next_follower1_q_values.max(dim=2)[0]
            next_follower2_q = next_follower2_q_values.max(dim=2)[0]
            
            # Compute target values
            leader_target = leader_rewards + (1 - dones) * self.gamma * next_leader_q
            follower1_target = follower1_rewards + (1 - dones) * self.gamma * next_follower1_q
            follower2_target = follower2_rewards + (1 - dones) * self.gamma * next_follower2_q
        
        # Compute loss for all agents
        leader_loss = nn.functional.mse_loss(leader_q, leader_target)
        follower1_loss = nn.functional.mse_loss(follower1_q, follower1_target)
        follower2_loss = nn.functional.mse_loss(follower2_q, follower2_target)
        
        # Optimize leader network
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1)  # Gradient clipping
        self.leader_optimizer.step()
        
        # Optimize follower1 network
        self.follower1_optimizer.zero_grad()
        follower1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 1)  # Gradient clipping
        self.follower1_optimizer.step()
        
        # Optimize follower2 network
        self.follower2_optimizer.zero_grad()
        follower2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 1)  # Gradient clipping
        self.follower2_optimizer.step()
        
        # Soft update target networks
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            self.soft_update(self.leader_online, self.leader_target)
            self.soft_update(self.follower1_online, self.follower1_target)
            self.soft_update(self.follower2_online, self.follower2_target)
            if self.use_qmix:
                self.soft_update(self.qmix_online, self.qmix_target)
        
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
        
        if self.use_qmix:
            torch.save(self.qmix_online.state_dict(), f"{path}/qmix_online.pt")
            torch.save(self.qmix_target.state_dict(), f"{path}/qmix_target.pt")
        
        params = {
            "epsilon": self.epsilon,
            "t_step": self.t_step,
            "hidden_size": self.hidden_size,
            "use_qmix": self.use_qmix
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
        
        if self.use_qmix and os.path.exists(f"{path}/qmix_online.pt"):
            self.qmix_online.load_state_dict(torch.load(f"{path}/qmix_online.pt", map_location=self.device))
            self.qmix_target.load_state_dict(torch.load(f"{path}/qmix_target.pt", map_location=self.device))
        
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
            self.epsilon = params["epsilon"]
            self.t_step = params["t_step"]
            # Check if the loaded model has the same configuration
            if "hidden_size" in params and params["hidden_size"] != self.hidden_size:
                print(f"Warning: Loaded model has hidden size {params['hidden_size']}, but current model has {self.hidden_size}")
            if "use_qmix" in params and params["use_qmix"] != self.use_qmix:
                print(f"Warning: Loaded model has use_qmix={params['use_qmix']}, but current model has use_qmix={self.use_qmix}")