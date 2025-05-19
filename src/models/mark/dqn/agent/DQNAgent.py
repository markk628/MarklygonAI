import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.types import Number

from src.config.config import (
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    DEVICE
)
from src.models.mark.dqn.model.DQNNetwork import DQNNetwork
from src.models.mark.dqn.model.DuelingDQNNetwork import DuelingDQNNetwork
from src.models.mark.dqn.utils.PrioritizedReplayBuffer import PrioritizedReplayBuffer

pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class DQNAgent:
    """
    Deep Q-Network agent with prioritized experience replay and dueling architecture option
    """
    def __init__(
        self, 
        feature_size: int,
        portfolio_info_size: int,
        market_info_size: int,
        constraint_size: int,
        action_size: int,
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
        use_prioritized: bool = True,
        per_alpha: float = 0.6, # Alpha for Prioritized Experience Replay
        per_beta: float = 0.4,  # Initial Beta for Prioritized Experience Replay
        per_beta_increment: float = 0.001 # Beta increment for PER
    ):
        self.feature_size: int = feature_size
        self.action_size: int = action_size
        self.total_steps: int = total_steps
        self.batch_size: int = batch_size
        self.discount_factor: float = discount_factor  # gamma (γ)
        self.epsilon: float = epsilon  # epsilon (ε)
        self.decay_rate_multiplier: float = decay_rate_multiplier
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay_target_pct: float = epsilon_decay_target_pct
        self.learning_rate: float = learning_rate
        self.use_dueling: bool = use_dueling
        self.use_prioritized: bool = use_prioritized
        self._current_step: int = 0

        # early forced exploration settings
        self.initial_exploration_episodes: int = 20
        self.force_trade_probability: float = 0.8
        self.current_episode: int = 0

        # device setup
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        # network initialization
        if use_dueling:
            self.main_network: DuelingDQNNetwork = DuelingDQNNetwork(feature_size, portfolio_info_size, market_info_size, constraint_size, action_size).to(self.device)
            self.target_network: DuelingDQNNetwork = DuelingDQNNetwork(feature_size, portfolio_info_size, market_info_size, constraint_size, action_size).to(self.device)
        else:
            self.main_network: DQNNetwork = DQNNetwork(feature_size, portfolio_info_size, market_info_size, constraint_size, action_size).to(self.device)
            self.target_network: DQNNetwork = DQNNetwork(feature_size, portfolio_info_size, market_info_size, constraint_size, action_size).to(self.device)
            
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
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        if self.use_prioritized:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True) -> Number:
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
                return random.randrange(self.action_size)

        # convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
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
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.stack([experience[0] for experience in minibatch]).astype(np.float32)
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.stack([experience[3] for experience in minibatch]).astype(np.float32)
            dones = np.array([experience[4] for experience in minibatch])

        # convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

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
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        # update priorities in buffer
        if self.use_prioritized:
            self.memory.update_priorities(indices, td_errors + 1e-6)  # small constant for stability

        # update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            
        # track loss
        self.loss_history.append(loss.item())

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = (self.epsilon_min) ** ((self._current_step / (self.total_steps * self.epsilon_decay_target_pct)) ** self.decay_rate_multiplier)
        # if self.epsilon > self.epsilon_min and self.total_steps * self.epsilon_decay_target_pct > 0:
        #     # Calculate the decay steps based on the target percentage
        #     decay_steps = self.total_steps * self.epsilon_decay_target_pct
        #     # Apply the decay formula
        #     self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.decay_rate_multiplier * self._current_step / decay_steps)

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