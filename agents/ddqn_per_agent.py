"""
Double Deep Q-Network with Prioritized Experience Replay for multi-robot coordination.

This module provides the DDQN agent implementation with prioritized experience replay 
for more efficient learning in the multi-robot battery disassembly task.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from agents.base_agent import BaseAgent, QMIXNetwork
from agents.replay_buffer import PrioritizedSequenceReplayBuffer


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network implementation for DDQN.
    
    This network separates state value and advantage streams for better value estimation.
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64):
        """
        Initialize the dueling Q-network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
        """
        super(DuelingQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )
        
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
            Q-values [batch_size, action_dim]
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Compute state value
        value = self.value_stream(features)
        
        # Compute action advantages
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class StackelbergDDQNPERAgent(BaseAgent):
    """
    Agent implementation using Double DQN with Prioritized Experience Replay for Stackelberg games with three robots.
    
    This agent uses double Q-learning to reduce overestimation bias and prioritized
    experience replay for more efficient learning.
    
    Attributes:
        hidden_size (int): Size of hidden layers
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate
        epsilon_decay (float): Rate at which epsilon decays over time
        epsilon_min (float): Minimum value for epsilon
        tau (float): Soft update parameter for target network
        update_every (int): How often to update the target network
        alpha (float): Priority exponent for PER
        beta (float): Initial importance sampling weight exponent for PER
        use_qmix (bool): Whether to use QMIX for mixing individual agent Q-values
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, action_dim_follower2: int,
                 hidden_size: int = 64, device: str = 'cpu', learning_rate: float = 1e-4,
                 gamma: float = 0.9, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 tau: float = 0.01, update_every: int = 10, alpha: float = 0.6, beta: float = 0.4,
                 use_qmix: bool = False, seed: int = 42, debug: bool = False):
        """
        Initialize the Stackelberg DDQN agent with PER for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            device: Device to run the model on (cpu or cuda)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays over time
            epsilon_min: Minimum value for epsilon
            tau: Soft update parameter for target network
            update_every: How often to update the target network
            alpha: Priority exponent for PER
            beta: Initial importance sampling weight exponent for PER
            use_qmix: Whether to use QMIX for mixing individual agent Q-values
            seed: Random seed
            debug: Whether to print debug information
        """
        super().__init__(state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, device, seed)
        
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_every = update_every
        self.alpha = alpha
        self.beta = beta
        self.use_qmix = use_qmix
        self.debug = debug
        
        # Initialize leader and follower networks
        self.leader_online = DuelingQNetwork(state_dim, action_dim_leader, hidden_size).to(device)
        self.leader_target = DuelingQNetwork(state_dim, action_dim_leader, hidden_size).to(device)
        
        self.follower1_online = DuelingQNetwork(state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower1_target = DuelingQNetwork(state_dim, action_dim_follower1, hidden_size).to(device)
        
        self.follower2_online = DuelingQNetwork(state_dim, action_dim_follower2, hidden_size).to(device)
        self.follower2_target = DuelingQNetwork(state_dim, action_dim_follower2, hidden_size).to(device)
        
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
        
        # Initialize training step counter
        self.t_step = 0
    
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
        
        # Get Q-values for all possible actions
        with torch.no_grad():
            leader_q_values = self.leader_online(state_tensor)
            follower1_q_values = self.follower1_online(state_tensor)
            follower2_q_values = self.follower2_online(state_tensor)
        
        # Apply action masks
        leader_q_values = self.apply_action_mask(leader_q_values, leader_mask)
        follower1_q_values = self.apply_action_mask(follower1_q_values, follower1_mask)
        follower2_q_values = self.apply_action_mask(follower2_q_values, follower2_mask)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()
        follower1_q = follower1_q_values.detach().cpu().numpy()
        follower2_q = follower2_q_values.detach().cpu().numpy()
        
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
    
    def update(self, experiences: Tuple[List[List[Tuple]], np.ndarray, np.ndarray]) -> Tuple[float, float, float]:
        """
        Update the Q-networks using a batch of experiences.
        
        Args:
            experiences: Tuple containing:
                - List of sequences of experience tuples
                - Indices of the sampled sequences
                - Importance sampling weights
                
        Returns:
            Losses for leader, follower1, and follower2
        """
        samples, indices, weights = experiences
        
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
        for sequence in samples:
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
        weights = torch.tensor(weights, dtype=torch.float).to(self.device).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1]
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Reshape for computing Q-values on individual steps
        flat_states = states.reshape(-1, self.state_dim)
        flat_next_states = next_states.reshape(-1, self.state_dim)
        
        # Compute current Q-values
        leader_q_values = self.leader_online(flat_states).reshape(batch_size, seq_len, -1)
        follower1_q_values = self.follower1_online(flat_states).reshape(batch_size, seq_len, -1)
        follower2_q_values = self.follower2_online(flat_states).reshape(batch_size, seq_len, -1)
        
        # Gather Q-values for the actions taken
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        leader_q = leader_q_values[batch_indices, seq_indices, leader_actions]
        follower1_q = follower1_q_values[batch_indices, seq_indices, follower1_actions]
        follower2_q = follower2_q_values[batch_indices, seq_indices, follower2_actions]
        
        # Double Q-learning target computation
        with torch.no_grad():
            # Select actions using online network
            next_leader_q_online = self.leader_online(flat_next_states).reshape(batch_size, seq_len, -1)
            next_follower1_q_online = self.follower1_online(flat_next_states).reshape(batch_size, seq_len, -1)
            next_follower2_q_online = self.follower2_online(flat_next_states).reshape(batch_size, seq_len, -1)
            
            next_leader_actions = next_leader_q_online.argmax(dim=2)
            next_follower1_actions = next_follower1_q_online.argmax(dim=2)
            next_follower2_actions = next_follower2_q_online.argmax(dim=2)
            
            # Evaluate actions using target network
            next_leader_q_target = self.leader_target(flat_next_states).reshape(batch_size, seq_len, -1)
            next_follower1_q_target = self.follower1_target(flat_next_states).reshape(batch_size, seq_len, -1)
            next_follower2_q_target = self.follower2_target(flat_next_states).reshape(batch_size, seq_len, -1)
            
            # Get Q-values for selected actions
            next_leader_q = next_leader_q_target.gather(2, next_leader_actions.unsqueeze(2)).squeeze(2)
            next_follower1_q = next_follower1_q_target.gather(2, next_follower1_actions.unsqueeze(2)).squeeze(2)
            next_follower2_q = next_follower2_q_target.gather(2, next_follower2_actions.unsqueeze(2)).squeeze(2)
            
            # Compute target values
            leader_target = leader_rewards + (1 - dones) * self.gamma * next_leader_q
            follower1_target = follower1_rewards + (1 - dones) * self.gamma * next_follower1_q
            follower2_target = follower2_rewards + (1 - dones) * self.gamma * next_follower2_q
        
        # Compute TD errors for PER
        td_error_leader = torch.abs(leader_q - leader_target).detach().cpu().numpy()
        td_error_follower1 = torch.abs(follower1_q - follower1_target).detach().cpu().numpy()
        td_error_follower2 = torch.abs(follower2_q - follower2_target).detach().cpu().numpy()
        
        # Compute mean TD error for each sequence
        mean_td_error = (np.mean(td_error_leader, axis=1) + 
                         np.mean(td_error_follower1, axis=1) + 
                         np.mean(td_error_follower2, axis=1))
        
        # Compute weighted MSE loss
        leader_loss = (weights * F.mse_loss(leader_q, leader_target.detach(), reduction='none')).mean()
        follower1_loss = (weights * F.mse_loss(follower1_q, follower1_target.detach(), reduction='none')).mean()
        follower2_loss = (weights * F.mse_loss(follower2_q, follower2_target.detach(), reduction='none')).mean()
        
        # Optimize networks
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1.0)
        self.leader_optimizer.step()
        
        self.follower1_optimizer.zero_grad()
        follower1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 1.0)
        self.follower1_optimizer.step()
        
        self.follower2_optimizer.zero_grad()
        follower2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 1.0)
        self.follower2_optimizer.step()
        
        # Update QMIX network if enabled
        if self.use_qmix:
            # Get agent Q-values for the actions taken
            agent_qs = [leader_q, follower1_q, follower2_q]
            agent_target_qs = [leader_target, follower1_target, follower2_target]
            
            # Reshape states for QMIX
            qmix_states = states.reshape(-1, self.state_dim)
            
            # Compute joint Q-values
            joint_q = self.qmix_online.get_joint_q_value(agent_qs, qmix_states)
            
            # Compute joint target Q-values
            with torch.no_grad():
                joint_target_q = self.qmix_target.get_joint_q_value(agent_target_qs, qmix_states)
            
            # Compute QMIX loss
            qmix_loss = (weights * F.mse_loss(joint_q, joint_target_q.detach(), reduction='none')).mean()
            
            # Optimize QMIX network
            self.qmix_optimizer.zero_grad()
            qmix_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qmix_online.parameters(), 1.0)
            self.qmix_optimizer.step()
        
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
        
        # Return the priorities
        return mean_td_error, leader_loss.item(), follower1_loss.item(), follower2_loss.item()
    
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