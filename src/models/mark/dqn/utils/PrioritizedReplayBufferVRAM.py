import numpy as np
import random
import torch

from src.config.config import DEVICE

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

import time

class PrioritizedReplayBufferVRAM:
    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = DEVICE
        
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        idx = self.position
        
        self.states[idx] = torch.from_numpy(state).to(self.device)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = torch.from_numpy(next_state).to(self.device)
        self.dones[idx] = done
        self.priorities[idx] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None, None, None
        
        # Priorities for available indices
        probs = (self.priorities[:self.size] + 1e-6) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs.cpu().numpy())
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        weights = (self.size * probs[indices]) ** -self.beta
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, priorities):
        indices = indices.to(self.device) if indices.device != self.device else indices
        priorities = priorities.to(self.device) if priorities.device != self.device else priorities

        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())

    def __len__(self):
        return self.size