import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from src.config.config import (
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    DEVICE
)
from src.models.mark.dqn.model.DQNNetwork import DQNNetwork
from src.models.mark.dqn.model.DuelingDQNNetwork import DuelingDQNNetwork
from src.models.mark.dqn.model.HierarchicalTradingDQNNetwork import HierarchicalTradingDQNNetwork
from src.models.mark.dqn.model.HierarchicalTradingDuelingDQNNetwork import HierarchicalTradingDuelingDQNNetwork
from src.models.mark.dqn.utils.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from src.models.mark.dqn.utils.PrioritizedReplayBufferVRAM import PrioritizedReplayBufferVRAM

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class DQNAgent:
    """
    Deep Q-Network agent with prioritized experience replay and dueling architecture option
    """
    def __init__(
        self, 
        sizes,
        total_steps: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        decay_rate_multiplier: float = 1,
        epsilon_min: float = 0.01,
        epsilon_decay_target_pct: float=1,
        batch_size: int = BATCH_SIZE,
        memory_size: int = REPLAY_BUFFER_SIZE,
        update_frequency: int = 4,
        target_update_frequency: int = 100,
        use_dueling: bool = True,
        use_hierarchical: bool = True,
        use_prioritized: bool = True,
        use_vram: bool = True,
        per_alpha: float = 0.6, # Alpha for Prioritized Experience Replay
        per_beta: float = 0.4,  # Initial Beta for Prioritized Experience Replay
        per_beta_increment: float = 0.001, # Beta increment for PER
        gradient_max_norm: float = 1.0
    ):
        self.sizes = sizes
        self.batch_size: int = batch_size
        self.discount_factor: float = discount_factor  # gamma (γ)
        self.epsilon: float = epsilon  # epsilon (ε)
        self.decay_rate_multiplier: float = decay_rate_multiplier
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay_target = total_steps * epsilon_decay_target_pct
        self.learning_rate: float = learning_rate
        self.use_dueling: bool = use_dueling
        self.use_prioritized: bool = use_prioritized
        self.use_vram: bool = use_vram
        self._current_step: int = 0

        # early forced exploration settings
        self.initial_exploration_episodes: int = 20
        self.force_trade_probability: float = 0.8
        self.current_episode: int = 0
        self._action_size = sizes['action_size']

        # device setup
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        # network initialization
        if use_dueling:
            if use_hierarchical:
                print('Using Hierarchical Dueling DQN')
                self.main_network: HierarchicalTradingDuelingDQNNetwork = HierarchicalTradingDuelingDQNNetwork(sizes).to(self.device)
                self.target_network: HierarchicalTradingDuelingDQNNetwork = HierarchicalTradingDuelingDQNNetwork(sizes).to(self.device)
            else:
                print('Using Dueling DQN')
                self.main_network: DuelingDQNNetwork = DuelingDQNNetwork(sizes).to(self.device)
                self.target_network: DuelingDQNNetwork = DuelingDQNNetwork(sizes).to(self.device)
        else:
            if use_hierarchical:
                print('Using Hierarchical DQN')
                self.main_network: HierarchicalTradingDQNNetwork = HierarchicalTradingDQNNetwork(sizes).to(self.device)
                self.target_network: HierarchicalTradingDQNNetwork = HierarchicalTradingDQNNetwork(sizes).to(self.device)
            else:
                print('Using DQN')
                self.main_network: DQNNetwork = DQNNetwork(sizes).to(self.device)
                self.target_network: DQNNetwork = DQNNetwork(sizes).to(self.device)
            
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() 

        # optimizer
        self.optimizer: optim.Adam = optim.AdamW(
            self.main_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5, 
            amsgrad=True
        )
        
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=3,
            verbose=True
        )
        
        # Memory setup
        if use_prioritized:
            if use_vram:
                stock_data_window_size = sizes['stock_data_window_size']
                stock_data_feature_size = sizes['stock_data_feature_size']
                stock_data_flattened_size = stock_data_window_size * stock_data_feature_size
                temporal_metrics_size = sizes['temporal_metrics_size'] * 3 if use_hierarchical else 0 # TODO if month and quarter gets added change the 3 to a 5
                state_dim = stock_data_flattened_size + sum(sizes.values()) - stock_data_window_size - stock_data_feature_size + temporal_metrics_size - self._action_size
                self.memory: PrioritizedReplayBufferVRAM = PrioritizedReplayBufferVRAM(memory_size, state_dim, alpha=per_alpha, beta=per_beta, beta_increment=per_beta_increment)
            else:
                self.memory: PrioritizedReplayBuffer = PrioritizedReplayBuffer(memory_size, alpha=per_alpha, beta=per_beta, beta_increment=per_beta_increment)
        else:
            self.memory: deque = deque(maxlen=memory_size)
            
        self.loss_fn: nn.SmoothL1Loss = nn.SmoothL1Loss()

        # Training parameters
        self.update_counter: int = 0
        self.target_update_frequency: int = target_update_frequency
        self.update_frequency: int = update_frequency
        self.training_steps: int = 0
        
        # Metrics tracking
        self.loss_history = []
        self.avg_q_values = []
        
        self.gradient_max_norm = gradient_max_norm
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        if self.use_prioritized:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True) -> int:
        """
        Select action using epsilon-greedy policy
        """
        # exploration during training
        if training:
            # early exploration stage
            if self.current_episode < self.initial_exploration_episodes:
                if random.random() < self.force_trade_probability:
                    # force buy or sell
                    return random.choice([0, 2])
            if np.random.rand() < self.epsilon:
                return random.randrange(self._action_size)

        # convert state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # get q-values from network
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(state)
        self.main_network.train()
        
        # track average q-values during training
        if training:
            self.avg_q_values.append(q_values.mean().item())
            
        return torch.argmax(q_values, dim=1).item()

    def train(self):
        """
        Train the agent by sampling from replay buffer
        """
        # skip if not enough samples
        if len(self.memory) < self.batch_size:
            return
                
        self.training_steps += 1
        
        # only update every update_frequency steps
        if self.training_steps % self.update_frequency != 0:
            return

        # sample from memory
        if self.use_prioritized:
            batch, indices, is_weights = self.memory.sample(self.batch_size)
            if batch is None:  # not enough samples
                return
                
            states, actions, rewards, next_states, dones = batch
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            batch = list(zip(*minibatch))
            states = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
            actions = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
            dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        # get current q-values
        q_values = self.main_network(states).gather(1, actions)

        # double DQN: get actions from main network
        with torch.no_grad():
            next_actions = self.main_network(next_states).max(1, keepdim=True)[1]
            # get q-values for those actions from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # calculate target q-values
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # calculate loss
        if self.use_prioritized:
            # TD errors for updating priorities
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            # wighted MSE loss
            loss = (is_weights.unsqueeze(1) * F.mse_loss(q_values, target_q_values, reduction='none')).mean()
        else:
            # SmoothL1Losss
            loss = self.loss_fn(q_values, target_q_values)
            
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.gradient_max_norm)
        self.optimizer.step()

        # update priorities in buffer
        if self.use_prioritized:
            if self.use_vram:
                td_errors = td_errors.squeeze()
            self.memory.update_priorities(indices, td_errors + 1e-6)  # small constant for stability

        # update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            
        # track loss
        self.loss_history.append(loss.item())

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = (self.epsilon_min) ** ((self._current_step / self.epsilon_decay_target) ** self.decay_rate_multiplier)
        # if self.epsilon > self.epsilon_min > 0:
        #     # Calculate the decay steps based on the target percentage
        #     # Apply the decay formula
        #     self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.decay_rate_multiplier * self._current_step / self.epsilon_decay_target)

        self._current_step += 1
        
    def load(self, file_path: str):
        """Load model weights from file"""
        self.main_network.load_state_dict(torch.load(file_path, map_location=self.device))
        self.target_network.load_state_dict(self.main_network.state_dict())
        print(f"Model loaded from {file_path}")

    def save(self, file_path: str):
        """Save model weights to file"""
        torch.save(self.main_network.state_dict(), file_path)
        print(f"Model saved to {file_path}")