import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, Dict
import random
from dataclasses import dataclass
import math
from datetime import datetime
from pathlib import Path

from src.config.config import (
    DEVICE,
    DATA_DIR,
    MODELS_DIR,
    WINDOW_SIZE,
    EVALUATE_INTERVAL,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    STOCK_FEATURES,
    NUM_EPISODES,
    TRAIN_RATIO,
    VALID_RATIO,
    TRAIN_INTERVAL,
)
from src.utils.utils import create_directory
from src.web.models import app, db, BacktestHistory, ModelType, MarklygonModel


@dataclass
class TradingConfig:
    """Configuration for the trading environment and DQN"""
    # Environment settings
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_pct: float = TRANSACTION_FEE_PERCENT
    max_position_size: float = 0.7
    
    # DQN settings
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001  # L2 regularization for AdamW
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 650000
    
    # PER settings
    per_alpha: float = 0.6  # Priority exponent
    per_beta_start: float = 0.4  # Importance sampling exponent
    per_beta_end: float = 1.0
    per_beta_decay: int = 100000
    per_epsilon: float = 0.001
    
    # Training settings
    batch_size: int = BATCH_SIZE
    buffer_size: int = REPLAY_BUFFER_SIZE
    update_target_every: int = 1000
    hidden_size: int = 512
    num_hidden_layers: int = 3
    
    # Features
    num_stock_features: int = len(STOCK_FEATURES)
    num_portfolio_features: int = 8
    num_features: int = num_stock_features + num_portfolio_features 
    window_size: int = WINDOW_SIZE  # WINDOW_SIZE minutes of historical data
    
    # 0 = Hold, 1 = Buy, 2 = Sell
    num_actions: int = 3


class DuelingNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams
    
    Architecture:
    1. Stock Data Branch (1D CNN)
    2. Portfolio Branch (Fully Connected)
    3. Combined (Value and Advantage Streams)
    """
    
    def __init__(self, config: TradingConfig):
        super(DuelingNetwork, self).__init__()
        self.config = config
        
        # Stock data branch - 1D CNN for time series processing
        # Input: (batch_size, window_size, self.config.num_stock_features)
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Portfolio state branch - Fully connected layers
        # Input: (batch_size, self.config.num_portfolio_features) - portfolio features
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Combined features: 256 (stock data branch output) + 64 (portfolio branch output) = 320
        combined_size = 256 + 64
        
        # Shared layers after combining branches
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU networks
        
        Weight initialization strategy:
        - He/Kaiming initialization for ReLU layers (prevents dying ReLU problem)
        - Small uniform initialization for output layers (stable Q-values)
        - Small positive bias for hidden layers (helps dead neurons)
        - Zero bias for output layers (unbiased initial predictions)
        
        This is crucial for DQN stability, especially in financial environments
        where poor initialization can lead to:
        - Overconfident initial predictions
        - Unstable training dynamics
        - Slow convergence
        """
        # Initialize stock data branch
        for layer in self.stock_data_branch:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                # He initialization (Kaiming) - best for ReLU networks
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize portfolio branch
        for layer in self.portfolio_branch:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize shared layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize value stream
        for layer in self.value_stream:
            if isinstance(layer, nn.Linear):
                if layer == self.value_stream[-1]:
                    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
                    nn.init.constant_(layer.bias, 0)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
        
        # Initialize advantage stream
        for layer in self.advantage_stream:
            if isinstance(layer, nn.Linear):
                if layer == self.advantage_stream[-1]:
                    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
                    nn.init.constant_(layer.bias, 0)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage streams
        
        Args:
            x: State tensor of shape (batch_size, window_size, self.config.num_features)
        """        
        # Split stock data and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]  # (batch_size, window_size, self.config.num_stock_features)
        portfolio_data = x[:, 0, self.config.num_stock_features:]  # (batch_size, self.config.num_portfolio_features)
        
        # Process stock data through CNN
        # Conv1d expects (batch_size, channels, window_size)
        stock_data = stock_data.permute(0, 2, 1)  # (batch_size, 40, window_size)
        stock_data_features = self.stock_data_branch(stock_data)  # (batch_size, 256, 1)
        stock_data_features = stock_data_features.squeeze(-1)  # (batch_size, 256)
        
        # Process portfolio data through FC layers
        portfolio_features = self.portfolio_branch(portfolio_data)  # (batch_size, 64)
        
        # Combine features
        combined_features = torch.cat([stock_data_features, portfolio_features], dim=1)  # (batch_size, 320)
        
        # Process through shared layers
        shared_features = self.shared_layers(combined_features)  # (batch_size, hidden_size)
        
        # Compute value and advantages
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBufferGPU:
    """PER VRAM version"""
    
    def __init__(self, capacity: int, config: TradingConfig, device: torch.device=DEVICE):
        self.capacity = capacity
        self.config = config
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate GPU tensors for the buffer
        self.states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # Priority management - initialize with small positive values
        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01  # Start with 0.01 instead of per_epsilon
        self.max_priority = 1.0
        
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool):
        """save experience"""
        # Ensure tensors are on the correct device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        # Store experience
        self.states[self.position] = state
        self.actions[self.position] = torch.tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.position] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.position] = next_state
        self.dones[self.position] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # Set priority to max for new experiences
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, beta: float) -> Tuple[torch.Tensor, ...]:
        """Sample batch of prioritizedexperiences"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        
        # Ensure priorities are positive
        priorities = torch.clamp(priorities, min=self.config.per_epsilon)
        
        # Calculate probabilities
        probs = priorities ** self.config.per_alpha
        probs_sum = probs.sum()
        
        # Handle edge case where sum is 0 or very small or contains NaN/inf
        if probs_sum < 1e-10 or torch.isnan(probs_sum) or torch.isinf(probs_sum):
            probs = torch.ones_like(probs) / self.size
        else:
            probs = probs / probs_sum
            
        # Additional check for NaN/inf in probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones(self.size, device=self.device) / self.size
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Ensure weights are valid
        weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
        weights = torch.clamp(weights, min=0.01, max=1.0)
        
        # Gather experiences
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities based on TD errors"""
        priorities = torch.abs(td_errors) + self.config.per_epsilon
        priorities = torch.clamp(priorities, min=max(self.config.per_epsilon, 0.001), max=1e6)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self):
        return self.size


class TradingMode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class TradingEnvironment:
    """Stock trading environment"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 scaled_data: pd.DataFrame, 
                 config: TradingConfig, 
                 mode: TradingMode = TradingMode.TRAIN, 
                 device: torch.device=DEVICE):
        self.data = data
        self.scaled_data = scaled_data
        self.config = config
        self.mode = mode
        self.total_steps = len(data)
        self.device = device
        
        self.reset()
        
    def reset(self, start_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment to initial state"""
        self.balance = self.config.initial_balance
        self.position = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        if start_idx is not None:
            self.current_step = start_idx
        elif self.mode == TradingMode.TRAIN:
            # Random start point for training
            max_start = max(self.config.window_size, len(self.data) - self.config.window_size - 100)
            if max_start > self.config.window_size:
                self.current_step = np.random.randint(self.config.window_size, max_start)
            else:
                self.current_step = self.config.window_size
        else:
            self.current_step = self.config.window_size
        self.total_steps = len(self.data) - self.current_step
            
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state"""
        # Get historical data
        start_idx = self.current_step - self.config.window_size + 1
        end_idx = self.current_step + 1
        stock_data = self.scaled_data.iloc[start_idx:end_idx].values
        
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + (self.position * current_price)
        
        # Normalize portfolio features (avoid division by zero)
        initial_balance = self.config.initial_balance
        normalized_balance = self.balance / initial_balance if initial_balance > 0 else 0
        normalized_position = self.position * current_price / initial_balance if initial_balance > 0 else 0
        normalized_portfolio_value = portfolio_value / initial_balance if initial_balance > 0 else 0
        
        # Calculate position ratio (0 if not holding, positive if long)
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0
        
        # Normalize other metrics
        normalized_trades = self.total_trades / (self.total_steps / 2)
        win_rate = self.winning_trades / max(1, self.total_trades)
        normalized_invalid_actions = self.invalid_actions / self.total_steps
        
        # Ensure all values are finite
        portfolio_features = [
            normalized_balance,
            normalized_position,
            normalized_portfolio_value,
            position_ratio,
            normalized_trades,
            win_rate,
            normalized_invalid_actions,
            1.0 if self.position > 0 else 0.0
        ]
        
        # Replace any non-finite values with 0
        portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
        
        # Create portfolio state vector
        portfolio_state = torch.tensor(np.array(portfolio_features), dtype=torch.float32, device=self.device)
        
        # Convert stock data to tensor
        stock_data_state = torch.tensor(stock_data, dtype=torch.float32, device=self.device)
        
        # Repeat portfolio state for each timestep and concatenate
        # This is needed for compatibility with the current state representation
        # The network will extract portfolio features from the first timestep
        portfolio_state_repeated = portfolio_state.unsqueeze(0).repeat(self.config.window_size, 1)
        
        # Concatenate stock data and portfolio features
        combined_state = torch.cat([stock_data_state, portfolio_state_repeated], dim=1)
        
        return combined_state
    
    def _is_invalid_action(self, action: int) -> bool:
        """Check if action is invalid"""
        current_price = self.data.iloc[self.current_step]['close']
        if action == 1:
            if self.position > 0 or self.balance <= 0:
                return True
            else:
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_pct)
                
                return cost > self.balance
        elif action == 2:
            return self.position == 0
        return False
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
            
        current_price = self.data.iloc[self.current_step]['close']        
        reward = 0
        trade_executed = False
        invalid_action = self._is_invalid_action(action)
        if invalid_action:
            self.invalid_actions += 1
            reward = -0.01
        else:
            if action == 1:  # Buy
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_pct)
                
                self.position = shares_to_buy
                self.balance -= cost
                self.entry_price = current_price
                trade_executed = True
                reward = 0.001
                    
            elif action == 2:  # Sell
                revenue = self.position * current_price * (1 - self.config.transaction_fee_pct)
                cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_pct)
                profit = revenue - cost_basis
                
                self.balance += revenue
                self.position = 0
                trade_executed = True
                self.total_trades += 1
                
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                percentage_return = profit / cost_basis
                reward = percentage_return * 10
        
            # Additional reward shaping
            # Penalize having too many invalid actions
            if self.invalid_actions > 100:
                reward -= 0.001 * (self.invalid_actions / 100)
            
            # Reward maintaining portfolio value
            current_portfolio_value = self.balance + (self.position * current_price)
            portfolio_return = (current_portfolio_value - self.config.initial_balance) / self.config.initial_balance
            
            # Small reward for positive returns
            if portfolio_return > 0:
                reward += 0.0001 * portfolio_return
            
            # Penalize if balance is getting too low
            if self.balance < self.config.initial_balance * 0.1:  # Less than 10% of initial
                reward -= 0.01
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 2 or self.balance <= 0
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'invalid_action': invalid_action,
            'invalid_actions': self.invalid_actions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
        
        return next_state, reward, done, info


class DoubleDuelingDQN:
    """Double Dueling DQN Agent with PER"""
    
    def __init__(self, config: TradingConfig, device: torch.device=DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        
        # Networks
        self.q_network = DuelingNetwork(config).to(device)
        self.target_network = DuelingNetwork(config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                    lr=config.learning_rate, 
                                    weight_decay=config.weight_decay)
        
        # Learning rate scheduler - reduces LR when validation reward plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        # Replay buffer
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config, device)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.update_count = 0
        
        
    def select_action(self, state: torch.Tensor, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if random.random() > epsilon:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.config.num_actions)
    
    def update(self) -> Dict[str, float]:
        """Perform one update step"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        try:
            # Calculate current beta for importance sampling
            beta = self.config.per_beta_start + (self.config.per_beta_end - self.config.per_beta_start) * min(1.0, self.steps_done / self.config.per_beta_decay)
            
            # Sample batch
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.config.batch_size, beta)
            
            # Compute current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Double DQN: use online network to select actions, target network to evaluate
            with torch.no_grad():
                next_actions = self.q_network(next_states).max(1)[1]
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
                target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
            
            # Compute TD errors for priority updates
            td_errors = (current_q_values - target_q_values).squeeze()
            
            # Clamp TD errors to prevent extreme values
            td_errors = torch.clamp(td_errors, min=-10.0, max=10.0)
            
            # Replace any NaN/inf values with zero
            td_errors = torch.where(torch.isfinite(td_errors), td_errors, torch.zeros_like(td_errors))
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors.detach())
            
            # Compute weighted loss
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none').squeeze()).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network
            self.update_count += 1
            if self.update_count % self.config.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            return {
                'loss': loss.item(),
                'mean_q': current_q_values.mean().item(),
                'mean_td_error': td_errors.abs().mean().item()
            }
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA Error in update: {e}")
                print(f"Memory size: {len(self.memory)}")
                print(f"Steps done: {self.steps_done}")
                # Try to recover by clearing CUDA cache
                torch.cuda.empty_cache()
                return {}
            else:
                raise
    
    def train_episode(self, env: TradingEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Track update metrics across the episode
        update_metrics = {
            'loss': [],
            'mean_q': [],
            'mean_td_error': []
        }
        
        while True:
            # Select and execute action
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, done)
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            self.steps_done += 1
            
            # Perform update only every TRAIN_INTERVAL steps
            if self.steps_done % TRAIN_INTERVAL == 0:
                update_info = self.update()
                # Track metrics if update occurred
                if update_info:
                    for key in ['loss', 'mean_q', 'mean_td_error']:
                        if key in update_info:
                            update_metrics[key].append(update_info[key])
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        self.episodes_done += 1
        
        # Episode statistics
        final_value = info['balance'] + (info['position'] * info['current_price'])
            
        # Calculate return
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        
        win_rate = info['winning_trades'] / max(1, info['total_trades'])
        
        # Average the update metrics over the episode
        avg_update_metrics = {}
        for key, values in update_metrics.items():
            if values:
                avg_update_metrics[key] = sum(values) / len(values)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': info['total_trades'],
            'win_rate': win_rate,
            'invalid_actions': info['invalid_actions'],
            **avg_update_metrics
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler state if available (for backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.update_count = checkpoint['update_count']

def load_stock_data(data_path: str, cutoff: pd.Timestamp | None=None, cols_to_keep: list[str]=STOCK_FEATURES) -> tuple[pd.DataFrame, datetime, datetime]:
    """
    Get saved csv data
    """
    df = pd.read_csv(data_path)
    start_date = cutoff.to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()

    if cutoff:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= cutoff]
        
    return df[cols_to_keep], start_date, end_date

def save_backtest_results_to_db(model_type: ModelType,
                                ticker: str,
                                info: dict[str, float],
                                preprocessor_path: Optional[str] = None) -> tuple[int, str, str]:
    backtest_date = info['backtest_date']
    return_rate = info['return_rate'] * 100
    
    with app.app_context():
        db.create_all()
        model = MarklygonModel(
            model=model_type,
            ticker=ticker
        )
        db.session.add(model)
        db.session.flush()
        
        model_id = model.id
        # Create directory for this model - use absolute path
        model_dir = str(MODELS_DIR / 'dqn_v2' / str(model_id))
        create_directory(model_dir)
        
        # Set model path within the model's directory - use absolute path
        model_path = str(Path(model_dir) / 'model.pth')
        model.model_path = model_path

        backtest = BacktestHistory(
            model=model,
            backtest_date=backtest_date,
            start_date=info['start_date'],
            end_date=info['end_date'],
            initial_balance=info['initial_balance'],
            final_balance=info['final_balance'],
            net_profit=info['net_profit'],
            total_trades=info['total_trades'],
            winning_trades=info['winning_trades'],
            losing_trades=info['losing_trades'],
            return_rate=return_rate,
            max_drawdown=info['max_drawdown'],
            sharpe_ratio=info['sharpe_ratio'],
            invalid_actions=info['invalid_actions'],
            preprocessor_path=preprocessor_path
        )

        db.session.add(backtest)
        db.session.commit()
    return model_id, model_path, model_dir

def train_dqn(data_path: str,
              cutoff: pd.Timestamp,
              num_episodes: int = NUM_EPISODES, 
              save_interval: int = 100,
              validation_frequency: int = EVALUATE_INTERVAL,
              early_stopping_patience: int = 10,
              train_ratio: float = TRAIN_RATIO,
              valid_ratio: float = VALID_RATIO,
              use_preprocessing: bool = True,
              scaling_method: str = 'robust',
              outlier_method: str = 'winsorize',
              preprocessor_save_path: Optional[str] = None):
    """
    Main training function with validation and early stopping
    
    Args:
        data_path: Path to CSV file with stock data
        cutoff: Timestamp to start data from
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        validation_frequency: Run validation every N episodes
        early_stopping_patience: Stop if validation doesn't improve for N checks
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        use_preprocessing: Whether to apply preprocessing
        scaling_method: Method for scaling features ('robust', 'standard', 'minmax', 'none')
        outlier_method: Method for handling outliers ('winsorize', 'clip', 'none')
        preprocessor_save_path: Path to save the fitted preprocessor
    """    
    # Load data
    print("Loading data...")
    data, start_date, end_date = load_stock_data(data_path, cutoff)
    
    # Split data chronologically
    train_end = int(len(data) * train_ratio)
    valid_end = train_end + int(len(data) * valid_ratio)
    
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    valid_data = data.iloc[train_end:valid_end].copy().reset_index(drop=True)
    test_data = data.iloc[valid_end:].copy().reset_index(drop=True)
    
    print(f"Data split: Train {len(train_data)}, Validation {len(valid_data)}, Test {len(test_data)}")
    
    # Apply preprocessing if requested
    preprocessor = None
    if use_preprocessing:
        print(f"\nApplying preprocessing (scaling: {scaling_method}, outliers: {outlier_method})...")
        from src.models.mark.dqn_v2.data_preprocessor import preprocess_financial_data
        
        # Set default preprocessor save path if not provided
        if preprocessor_save_path is None:
            from pathlib import Path
            data_name = Path(data_path).stem
            preprocessor_save_path = f"preprocessor_{data_name}.pkl"
        
        # Preprocess data
        preprocessor, train_data_scaled, valid_data_scaled, test_data_scaled = preprocess_financial_data(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            scaling_method=scaling_method,
            outlier_method=outlier_method,
            save_preprocessor=True,
            preprocessor_path=preprocessor_save_path
        )
        print(f"Preprocessing complete. Preprocessor saved to: {preprocessor_save_path}")
    else:
        # If no preprocessing, use raw data
        train_data_scaled = train_data
        valid_data_scaled = valid_data
        test_data_scaled = test_data
    
    # Initialize configuration
    config = TradingConfig()
    
    # Create environments
    train_env = TradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = TradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = TradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    # Create agent
    agent = DoubleDuelingDQN(config)
    
    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    episode_invalid_actions = []
    
    validation_rewards = []
    validation_returns = []
    validation_trades = []
    validation_invalid_actions = []
    
    best_validation_return = float('-inf')
    patience_counter = 0
    best_model_state = None
    
    print("Starting training...")
    
    for episode in range(num_episodes):
        # Train one episode
        metrics = agent.train_episode(train_env)
        
        # Store metrics
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        episode_trades.append(metrics['total_trades'])
        episode_invalid_actions.append(metrics['invalid_actions'])
        
        # Print progress
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {metrics['episode_reward']:.4f}")
        print(f"  Return: {metrics['total_return']:.2%}")
        print(f"  Final Value: ${metrics['final_value']:,.2f}")
        print(f"  Trades: {metrics['total_trades']}, Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Invalid Actions: {metrics['invalid_actions']}")
        print(f"  Steps: {metrics['episode_steps']}")
        print(f"  Epsilon: {agent.config.epsilon_end + (agent.config.epsilon_start - agent.config.epsilon_end) * math.exp(-1. * agent.steps_done / agent.config.epsilon_decay):.4f}")
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print("\nRunning validation...")
            
            # Run validation episode
            val_state = val_env.reset()
            val_reward = 0
            val_done = False
            
            while not val_done:
                val_action = agent.select_action(val_state, epsilon=0.0)  # No exploration
                val_next_state, val_r, val_done, val_info = val_env.step(val_action)
                val_reward += val_r
                val_state = val_next_state
            
            val_final_value = val_info['balance'] + (val_info['position'] * val_info['current_price'])
            val_return = (val_final_value - config.initial_balance) / config.initial_balance
            
            validation_rewards.append(val_reward)
            validation_returns.append(val_return)
            validation_trades.append(val_info['total_trades'])
            validation_invalid_actions.append(val_info['invalid_actions'])
            
            print(f"Validation Results:")
            print(f"  Return: {val_return:.2%}")
            print(f"  Final Value: ${val_final_value:,.2f}")
            print(f"  Trades: {val_info['total_trades']}")
            print(f"  Win Rate: {val_info['winning_trades'] / max(1, val_info['total_trades']):.2%}")
            print(f"  Invalid Actions: {val_info['invalid_actions']}")
            
            # Update learning rate scheduler based on validation return
            agent.scheduler.step(val_return)
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
            
            # Early stopping check
            if val_return > best_validation_return:
                best_validation_return = val_return
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    'q_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'scheduler': agent.scheduler.state_dict(),
                    'steps_done': agent.steps_done,
                    'episodes_done': agent.episodes_done,
                    'update_count': agent.update_count
                }
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at episode {episode+1}")
                print(f"Best validation return: {best_validation_return:.2%}")
                
                # Restore best model
                if best_model_state:
                    agent.q_network.load_state_dict(best_model_state['q_network'])
                    agent.target_network.load_state_dict(best_model_state['target_network'])
                    agent.optimizer.load_state_dict(best_model_state['optimizer'])
                    agent.scheduler.load_state_dict(best_model_state['scheduler'])
                    agent.steps_done = best_model_state['steps_done']
                    agent.episodes_done = best_model_state['episodes_done']
                    agent.update_count = best_model_state['update_count']
                break
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            agent.save(f"dqn_checkpoint_episode_{episode}.pt")
            print(f"Saved checkpoint at episode {episode}")
    
    # Final test evaluation
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    
    test_state = test_env.reset()
    test_reward = 0
    test_done = False
    test_action_history = []
    test_portfolio_values = [config.initial_balance]
    test_price_history = []
    
    while not test_done:
        test_action = agent.select_action(test_state, epsilon=0.0)
        test_next_state, test_r, test_done, test_info = test_env.step(test_action)
        test_reward += test_r
        test_state = test_next_state
        
        # Track for plotting - only append valid actions
        if not test_info['invalid_action']:
            test_action_history.append(test_action)
        current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
        test_portfolio_values.append(current_value)
        test_price_history.append(test_info['current_price'])
    
    # Calculate final metrics
    test_final_value = test_portfolio_values[-1]
    test_return = (test_final_value - config.initial_balance) / config.initial_balance
    
    # Calculate Sharpe ratio
    returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
    sharpe = np.sqrt(252 * 390) * returns.mean() / (returns.std() + 1e-10)
    
    # Calculate max drawdown
    peak = np.maximum.accumulate(test_portfolio_values)
    drawdown = (test_portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    print(f"\nTest Results:")
    print(f"  Initial Balance: ${config.initial_balance:,.2f}")
    print(f"  Final Value: ${test_final_value:,.2f}")
    print(f"  Total Return: {test_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Total Trades: {test_info['total_trades']}")
    print(f"  Win Rate: {test_info['winning_trades'] / max(1, test_info['total_trades']):.2%}")
    print(f"  Invalid Actions: {test_info['invalid_actions']}")
    
    # Return comprehensive results
    return {
        'agent': agent,
        'preprocessor': preprocessor,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_invalid_actions': episode_invalid_actions,
        'validation_rewards': validation_rewards,
        'validation_returns': validation_returns,
        'validation_trades': validation_trades,
        'validation_invalid_actions': validation_invalid_actions,
        'test_results': {
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'win_rate': test_info['winning_trades'] / max(1, test_info['total_trades']),
            'invalid_actions': test_info['invalid_actions'],
            'action_history': test_action_history,
            'portfolio_values': test_portfolio_values,
            'price_history': test_price_history
        },
        'start_date': start_date,
        'end_date': end_date
    }


if __name__ == "__main__":
    train_dqn(f"{DATA_DIR}/feature_engineered/TSLA.csv", 
              num_episodes=50,
              use_preprocessing=True,
              scaling_method='robust',
              outlier_method='winsorize')
