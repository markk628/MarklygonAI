import numpy as np
import random
import torch

from src.config.config import DEVICE

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay for more efficient learning
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = beta  # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # Beta increases over time for more correction
        self.device = DEVICE
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # Initial max priority for new transitions
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a new experience with max priority
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # New experiences get max priority to ensure they're sampled
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample experiences based on their priorities
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = (priorities + 1e-6) ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get samples and calculate importance sampling weights
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices] + 1e-6) ** -self.beta
        weights /= weights.max()  # Normalize weights
        
        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(map(list, zip(*samples)))

        states = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities based on TD errors
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return len(self.buffer)