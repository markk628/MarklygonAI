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

# Import the replay buffer and environment from DQN
from src.models.mark.dqn_v2.dqn import TradingMode, load_stock_data, save_backtest_results_to_db


@dataclass
class ContinuousSACConfig:
    """Configuration for the continuous SAC trading agent"""
    # Environment settings
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_pct: float = TRANSACTION_FEE_PERCENT
    max_position_ratio: float = 1.0  # Maximum position as ratio of portfolio value
    
    # SAC settings
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    tau: float = 0.005
    
    # Entropy settings
    target_entropy: float = -1.0  # For continuous actions
    alpha_auto_tune: bool = True
    
    # PER settings
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_decay: int = 100000
    per_epsilon: float = 0.001
    
    # Training settings
    batch_size: int = BATCH_SIZE
    buffer_size: int = REPLAY_BUFFER_SIZE
    hidden_size: int = 512
    
    # Features
    num_stock_features: int = 39
    num_portfolio_features: int = 8
    num_features: int = num_stock_features + num_portfolio_features 
    window_size: int = WINDOW_SIZE
    
    # Action space: continuous [-1, 1] where:
    # -1 = sell all holdings, 0 = hold, +1 = buy maximum amount
    action_dim: int = 1


class ContinuousActorNetwork(nn.Module):
    """Continuous policy network that outputs position sizing actions"""
    
    def __init__(self, config: ContinuousSACConfig):
        super(ContinuousActorNetwork, self).__init__()
        self.config = config
        
        # Stock data branch - 1D CNN
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
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
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
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
        
        # Combined features
        combined_size = 256 + 64
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Mean and log_std heads for continuous actions
        self.mean_head = nn.Linear(config.hidden_size, config.action_dim)
        self.log_std_head = nn.Linear(config.hidden_size, config.action_dim)
        
        # Action bounds
        self.action_scale = 1.0  # Actions in [-1, 1]
        self.action_bias = 0.0
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.stock_data_branch, self.portfolio_branch, self.shared_layers]:
            for layer in module:
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
        
        # Initialize output layers with smaller weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std for action distribution"""
        # Split input
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # Process branches
        stock_data = stock_data.permute(0, 2, 1)
        stock_features = self.stock_data_branch(stock_data).squeeze(-1)
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine and process
        combined = torch.cat([stock_features, portfolio_features], dim=1)
        shared_features = self.shared_layers(combined)
        
        # Get mean and log_std
        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent extreme values
        
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from the policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)
            
        # Apply scaling
        action = action * self.action_scale + self.action_bias
        
        return action
    
    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and its log probability"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Compute log probability with change of variables formula
            log_prob = normal.log_prob(x_t)
            # Correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Apply scaling
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob


class ContinuousCriticNetwork(nn.Module):
    """Q-value network for continuous actions"""
    
    def __init__(self, config: ContinuousSACConfig):
        super(ContinuousCriticNetwork, self).__init__()
        self.config = config
        
        # Stock data branch
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
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
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
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
        
        # Q-value head (state + action -> Q-value)
        combined_size = 256 + 64 + config.action_dim  # state features + action
        self.q_head = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.stock_data_branch, self.portfolio_branch, self.q_head]:
            for layer in module:
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value for state-action pair"""
        # Split state input
        stock_data = state[:, :, :self.config.num_stock_features]
        portfolio_data = state[:, 0, self.config.num_stock_features:]
        
        # Process state branches
        stock_data = stock_data.permute(0, 2, 1)
        stock_features = self.stock_data_branch(stock_data).squeeze(-1)
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine state features with action
        state_features = torch.cat([stock_features, portfolio_features], dim=1)
        state_action = torch.cat([state_features, action], dim=1)
        
        # Compute Q-value
        q_value = self.q_head(state_action)
        
        return q_value


class ContinuousPrioritizedReplayBuffer:
    """Replay buffer for continuous actions"""
    
    def __init__(self, capacity: int, config: ContinuousSACConfig, device: torch.device = DEVICE):
        self.capacity = capacity
        self.config = config
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, config.action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # Priority management
        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01
        self.max_priority = 1.0
    
    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        """Add experience to buffer"""
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.position] = next_state
        self.dones[self.position] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float) -> Tuple[torch.Tensor, ...]:
        """Sample batch with importance sampling weights"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        priorities = self.priorities[:self.size]
        priorities = torch.clamp(priorities, min=self.config.per_epsilon)
        
        probs = priorities ** self.config.per_alpha
        probs_sum = probs.sum()
        
        if probs_sum < 1e-10 or torch.isnan(probs_sum) or torch.isinf(probs_sum):
            probs = torch.ones_like(probs) / self.size
        else:
            probs = probs / probs_sum
        
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones(self.size, device=self.device) / self.size
        
        try:
            indices = torch.multinomial(probs, batch_size, replacement=True)
        except RuntimeError:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
        weights = torch.clamp(weights, min=0.01, max=1.0)
        
        return (self.states[indices], self.actions[indices], self.rewards[indices], 
                self.next_states[indices], self.dones[indices], indices, weights)
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities based on TD errors"""
        priorities = torch.abs(td_errors) + self.config.per_epsilon
        priorities = torch.clamp(priorities, min=0.001, max=1e6)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self):
        return self.size


class ContinuousTradingEnvironment:
    """Trading environment for continuous actions"""
    
    def __init__(self, data: pd.DataFrame, scaled_data: pd.DataFrame, config: ContinuousSACConfig, 
                 mode: TradingMode = TradingMode.TRAIN, device: torch.device = DEVICE):
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
        self.position_value = 0.0  # Current position value in dollars
        self.shares_held = 0.0  # Number of shares held
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        self.total_fees_paid = 0.0
        
        # Set starting position
        if start_idx is not None:
            self.current_step = start_idx
        elif self.mode == TradingMode.TRAIN:
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
        """Get current state tensor"""
        # Get historical data
        start_idx = self.current_step - self.config.window_size + 1
        end_idx = self.current_step + 1
        stock_data = self.scaled_data.iloc[start_idx:end_idx].values
        
        # Calculate portfolio metrics
        current_price = self.data.iloc[self.current_step]['close']
        current_position_value = self.shares_held * current_price
        total_portfolio_value = self.balance + current_position_value
        
        # Normalize portfolio features with safe division
        initial_balance = self.config.initial_balance
        normalized_balance = float(self.balance / initial_balance if initial_balance > 0 else 0.0)
        normalized_position_value = float(current_position_value / initial_balance if initial_balance > 0 else 0.0)
        normalized_total_value = float(total_portfolio_value / initial_balance if initial_balance > 0 else 0.0)
        
        # Position ratio (what fraction of portfolio is in stocks)
        position_ratio = float(current_position_value / total_portfolio_value if total_portfolio_value > 0 else 0.0)
        
        # Other metrics with safe division
        normalized_trades = float(self.total_trades / max(1, self.total_steps / 10))
        win_rate = float(self.winning_trades / max(1, self.total_trades))
        normalized_fees = float(self.total_fees_paid / initial_balance if initial_balance > 0 else 0.0)
        
        # Create portfolio features list with explicit float conversion
        portfolio_features = [
            normalized_balance,
            normalized_position_value,
            normalized_total_value,
            position_ratio,
            normalized_trades,
            win_rate,
            normalized_fees,
            float(1.0 if self.shares_held > 0 else 0.0)
        ]
        
        # Ensure all values are finite and convert to float
        portfolio_features = [float(x) if np.isfinite(float(x)) else 0.0 for x in portfolio_features]
        
        # Verify we have the expected number of features
        assert len(portfolio_features) == self.config.num_portfolio_features, \
            f"Expected {self.config.num_portfolio_features} portfolio features, got {len(portfolio_features)}"
        
        # Create state tensors with explicit dtype conversion
        portfolio_state = torch.tensor(portfolio_features, dtype=torch.float32, device=self.device)
        stock_data_state = torch.tensor(stock_data, dtype=torch.float32, device=self.device)
        
        # Repeat portfolio state and concatenate
        portfolio_state_repeated = portfolio_state.unsqueeze(0).repeat(self.config.window_size, 1)
        combined_state = torch.cat([stock_data_state, portfolio_state_repeated], dim=1)
        
        return combined_state
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute continuous action and return results"""
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Convert action to target position ratio
        # action is in [-1, 1] where:
        # -1 = sell all (target position ratio = 0)
        # 0 = hold current position
        # +1 = buy maximum (target position ratio = max_position_ratio)
        action_value = action.item() if torch.is_tensor(action) else action
        action_value = np.clip(action_value, -1.0, 1.0)
        
        # Calculate current portfolio value
        current_position_value = self.shares_held * current_price
        total_portfolio_value = self.balance + current_position_value
        current_position_ratio = current_position_value / total_portfolio_value if total_portfolio_value > 0 else 0
        
        # Calculate target position ratio
        if action_value >= 0:
            # Buying: scale from current ratio to max ratio
            target_position_ratio = current_position_ratio + action_value * (self.config.max_position_ratio - current_position_ratio)
        else:
            # Selling: scale from current ratio to 0
            target_position_ratio = current_position_ratio * (1 + action_value)
        
        target_position_ratio = np.clip(target_position_ratio, 0, self.config.max_position_ratio)
        
        # Calculate target position value and required action
        target_position_value = target_position_ratio * total_portfolio_value
        position_change = target_position_value - current_position_value
        
        reward = 0
        trade_executed = False
        invalid_action = False
        
        # Execute the trade if significant enough
        if abs(position_change) > total_portfolio_value * 0.01:  # 1% threshold
            if position_change > 0:  # Buying
                max_buyable = self.balance / (current_price * (1 + self.config.transaction_fee_pct))
                shares_to_buy = min(position_change / current_price, max_buyable)
                
                if shares_to_buy > 0.001:  # Minimum share threshold
                    cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_pct)
                    if cost <= self.balance:
                        self.shares_held += shares_to_buy
                        self.balance -= cost
                        self.total_fees_paid += shares_to_buy * current_price * self.config.transaction_fee_pct
                        trade_executed = True
                        reward = 0.001  # Small reward for valid trade
                    else:
                        invalid_action = True
                        reward = -0.01
            
            else:  # Selling
                shares_to_sell = min(abs(position_change) / current_price, self.shares_held)
                
                if shares_to_sell > 0.001:  # Minimum share threshold
                    revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_pct)
                    
                    # Calculate profit/loss for this trade
                    avg_cost_per_share = (self.config.initial_balance - self.balance + self.total_fees_paid) / max(self.shares_held, 1e-6)
                    profit = (current_price - avg_cost_per_share) * shares_to_sell
                    
                    self.shares_held -= shares_to_sell
                    self.balance += revenue
                    self.total_fees_paid += shares_to_sell * current_price * self.config.transaction_fee_pct
                    self.total_trades += 1
                    trade_executed = True
                    
                    if profit > 0:
                        self.winning_trades += 1
                        reward = profit / self.config.initial_balance * 10  # Scale reward
                    else:
                        self.losing_trades += 1
                        reward = profit / self.config.initial_balance * 10
        
        # Additional reward shaping
        if not invalid_action:
            # Reward for maintaining portfolio value
            new_position_value = self.shares_held * current_price
            new_total_value = self.balance + new_position_value
            portfolio_return = (new_total_value - self.config.initial_balance) / self.config.initial_balance
            
            if portfolio_return > 0:
                reward += 0.0001 * portfolio_return
            
            # Penalty for too many fees
            fee_ratio = self.total_fees_paid / self.config.initial_balance
            if fee_ratio > 0.05:  # More than 5% in fees
                reward -= 0.001 * fee_ratio
        
        if invalid_action:
            self.invalid_actions += 1
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 2 or (self.balance <= 0 and self.shares_held <= 0)
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position_value': self.shares_held * current_price,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'invalid_action': invalid_action,
            'invalid_actions': self.invalid_actions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_fees_paid': self.total_fees_paid,
            'target_position_ratio': target_position_ratio,
            'action_value': action_value
        }
        
        return next_state, reward, done, info


# Export the classes
__all__ = [
    'ContinuousSACConfig',
    'ContinuousActorNetwork', 
    'ContinuousCriticNetwork',
    'ContinuousPrioritizedReplayBuffer',
    'ContinuousTradingEnvironment'
] 