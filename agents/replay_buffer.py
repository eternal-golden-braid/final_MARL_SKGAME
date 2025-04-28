"""
Replay buffer implementations for sequence-based reinforcement learning.

This module provides replay buffer classes for storing and sampling
experiences, especially for recurrent neural networks that require sequence data.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union


class SequenceReplayBuffer:
    """
    Replay buffer for storing and sampling sequences of experiences.
    
    This buffer is designed for recurrent agents that need to maintain
    temporal dependencies in their learning.
    
    Attributes:
        buffer_size (int): Maximum size of the buffer
        sequence_length (int): Length of each sequence
        batch_size (int): Default batch size for sampling
        rng (np.random.Generator): Random number generator
        buffer (list): List to store sequences
        episode_buffer (list): Buffer for the current episode
    """
    def __init__(self, buffer_size: int, sequence_length: int, batch_size: int, seed: int = 42):
        """
        Initialize the sequence replay buffer.
        
        Args:
            buffer_size: Maximum number of sequences to store
            sequence_length: Length of each sequence
            batch_size: Default batch size for sampling
            seed: Random seed
        """
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.episode_buffer = []
        # Keep track of a unique zero state for padding
        self.zero_state = None
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)
    
    def add(self, experience: Tuple) -> None:
        """
        Add an experience to the episode buffer.
        
        Args:
            experience: Experience to add [state, a_leader, a_follower1, a_follower2, 
                                         r_leader, r_follower1, r_follower2, next_state, done]
        """
        self.episode_buffer.append(experience)
    
    def end_episode(self, zero_state: Optional[np.ndarray] = None) -> None:
        """
        End the current episode and transfer sequences to the main buffer.
        
        Args:
            zero_state: Zero state to use for padding (if None, use zeros of the same shape as states)
        """
        if len(self.episode_buffer) == 0:
            return
        
        # If zero_state not provided, create one with zeros of the same shape as the state
        if zero_state is None:
            if self.zero_state is None:
                # Create a zero state based on the first state in the episode
                # This ensures consistent shape for all padding
                first_state = self.episode_buffer[0][0]
                self.zero_state = np.zeros_like(first_state)
        else:
            self.zero_state = zero_state
        
        # Add overlapping sequences from the episode to the buffer
        for i in range(max(1, len(self.episode_buffer) - self.sequence_length + 1)):
            sequence = self.episode_buffer[i:i+self.sequence_length]
            if len(sequence) < self.sequence_length:
                # Pad shorter sequences with custom zero-state padding
                if self.zero_state is not None:
                    # Create padding experiences with the zero state
                    padding = []
                    for _ in range(self.sequence_length - len(sequence)):
                        # State, actions (-1 for no action), rewards (0), next_state, done=False
                        padding_exp = (
                            self.zero_state,  # State
                            -1,  # Leader action
                            -1,  # Follower1 action
                            -1,  # Follower2 action
                            0.0,  # Leader reward
                            0.0,  # Follower1 reward
                            0.0,  # Follower2 reward
                            self.zero_state,  # Next state
                            False  # Not done
                        )
                        padding.append(padding_exp)
                else:
                    # Fallback: clone last experience but set done=False
                    last_exp = list(sequence[-1])
                    last_exp[-1] = False  # Set done flag to False
                    padding = [tuple(last_exp)] * (self.sequence_length - len(sequence))
                
                sequence.extend(padding)
            
            self.buffer.append(sequence)
            
            # Maintain buffer size
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        
        self.episode_buffer = []
    
    def sample(self, batch_size: Optional[int] = None) -> List:
        """
        Sample a batch of sequences from the buffer.
        
        Args:
            batch_size: Size of batch to sample (uses default if None)
        
        Returns:
            List of sequence experiences
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} sequences, but requested batch size is {batch_size}")
        
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class PrioritizedSequenceReplayBuffer(SequenceReplayBuffer):
    """
    Prioritized Experience Replay buffer for storing sequences of experiences.
    
    This buffer samples experiences based on their TD error, giving
    higher priority to experiences with higher error.
    
    Attributes:
        alpha (float): Priority exponent
        beta (float): Importance sampling weight exponent
        beta_increment (float): Beta increment per sampling
        priorities (np.ndarray): Array of priorities
    """
    def __init__(self, buffer_size: int, sequence_length: int, batch_size: int, 
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, seed: int = 42):
        """
        Initialize the prioritized sequence replay buffer.
        
        Args:
            buffer_size: Maximum number of sequences to store
            sequence_length: Length of each sequence
            batch_size: Default batch size for sampling
            alpha: Priority exponent
            beta: Importance sampling weight exponent
            beta_increment: Beta increment per sampling
            seed: Random seed
        """
        super().__init__(buffer_size, sequence_length, batch_size, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
    
    def end_episode(self, zero_state: Optional[np.ndarray] = None) -> None:
        """
        End the current episode and transfer sequences to the main buffer.
        
        Args:
            zero_state: Zero state to use for padding (if None, use the last experience)
        """
        prev_size = len(self.buffer)
        super().end_episode(zero_state)
        
        # Initialize priorities for new sequences with max priority
        max_priority = 1.0 if prev_size == 0 else np.max(self.priorities[:prev_size])
        for i in range(prev_size, len(self.buffer)):
            idx = i % self.buffer_size
            self.priorities[idx] = max_priority
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample a batch of sequences from the buffer based on priorities.
        
        Args:
            batch_size: Size of batch to sample (uses default if None)
        
        Returns:
            Tuple containing:
            - List of sequence experiences
            - Indices of the sampled sequences
            - Importance sampling weights
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} sequences, but requested batch size is {batch_size}")
        
        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Sample indices based on probabilities
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        
        # Increment beta for next sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get sampled experiences
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities of experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx % self.buffer_size] = priority + 1e-5  # Add small value to avoid zero priority