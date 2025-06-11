"""
SAC v1: Soft Actor-Critic Trading Agent with Continuous Action Space
==================================================================

Features:
- Continuous action space for precise buy/sell amounts
- Same reward system as DQN v5 for consistency
- GPU-based replay buffer for efficiency
- Portfolio state normalization with warmup
- Enhanced financial architectures adapted for actor-critic
- Automatic entropy tuning (SAC-AET)
- Twin critics for stability (similar to TD3)

Action Space:
- Single continuous value in [-1, 1]
- Negative values: sell (magnitude = proportion of holdings to sell)
- Positive values: buy (magnitude = proportion of available cash to use)
- 0: hold (no action)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import random
import math
from datetime import datetime, time
from enum import Enum
from pathlib import Path

from src.config.config import (
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    MAX_POSITION_SIZE,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    DEVICE,
    EPSILON_EARLY_STOPPING_THRESHOLD,
    STOCK_FEATURES_V2,
    NUM_EPISODES,
    TRAIN_RATIO,
    VALID_RATIO,
    EVALUATE_INTERVAL,
    MINUTES_PER_TRADING_DAY
)

from src.models.mark.dqn_v2.normalization import PortfolioStateNormalizer
from src.models.mark.dqn_v2.networks.base import FinancialTransformerBlock


class SACConfig:
    """Configuration for SAC agent"""
    # Environment parameters
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    window_size: int = WINDOW_SIZE
    num_stock_features: int = len(STOCK_FEATURES_V2)
    num_portfolio_features: int = 13
    num_features: int = len(STOCK_FEATURES_V2) + 13
    max_position_size: float = MAX_POSITION_SIZE
    
    # Network parameters
    actor_hidden_size: int = 512
    critic_hidden_size: int = 512
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    
    # Training parameters
    batch_size: int = BATCH_SIZE
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    update_frequency: int = 1
    
    # Experience replay
    buffer_size: int = REPLAY_BUFFER_SIZE
    
    # SAC specific parameters
    target_entropy: float = -0.5  # Less conservative than -1.0 for more active trading
    alpha_auto_tune: bool = True  # Re-enabled for proper SAC learning
    initial_alpha: float = 0.1  # Starting value, will be auto-tuned
    
    # Portfolio state normalization
    use_portfolio_normalization: bool = True
    portfolio_warmup_episodes: int = 50
    portfolio_update_frequency: int = 100
    
    # Trading parameters
    min_trade_amount: float = 0.001  # Minimum 0.1% position size for trades (reduced from 1%)
    
    # Reward parameters (optimizable)
    portfolio_scaling: float = 1.0  # Increased from 0.1 to make trading more rewarding
    invalid_penalty: float = 0.01  # Reduced penalty for invalid actions
    
    # PER parameters
    per_alpha: float = 0.6  # Prioritization strength
    per_beta_start: float = 0.4  # Initial importance sampling
    per_beta_end: float = 1.0  # Final importance sampling
    per_epsilon: float = 0.001  # Small constant for numerical stability
    
    def __post_init__(self):
        # Set target entropy automatically for 1D action space
        # Less conservative than -1.0 to encourage more active trading
        if self.target_entropy == -1.0:
            self.target_entropy = -0.5  # For 1D action space, more active than standard -1.0


class PrioritizedReplayBufferGPU:
    """Prioritized Experience Replay buffer stored on GPU for SAC"""
    
    def __init__(self, capacity: int, config: SACConfig, device: torch.device = DEVICE):
        self.capacity = capacity
        self.config = config
        self.device = device
        self.position = 0
        self.size = 0
        
        # Store experiences on GPU
        self.states = torch.zeros((capacity, config.window_size, config.num_features), 
                                 dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), 
                                      dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # PER parameters
        self.alpha = getattr(config, 'per_alpha', 0.6)  # Prioritization strength
        self.beta_start = getattr(config, 'per_beta_start', 0.4)  # Initial importance sampling
        self.beta_end = getattr(config, 'per_beta_end', 1.0)  # Final importance sampling
        self.per_epsilon = getattr(config, 'per_epsilon', 0.001)  # Small constant for numerical stability
        
        # Priority storage
        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01
        self.max_priority = 1.0
        
    def push(self, state: torch.Tensor, action: float, reward: float, 
             next_state: torch.Tensor, done: bool):
        """Store experience with maximum priority for new experiences"""
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        self.states[self.position] = state
        self.actions[self.position] = torch.tensor([[action]], dtype=torch.float32, device=self.device)
        self.rewards[self.position] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.position] = next_state
        self.dones[self.position] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # Set priority to max for new experiences
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, beta: float) -> Tuple[torch.Tensor, ...]:
        """Sample batch of prioritized experiences"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        
        # Ensure priorities are positive
        priorities = torch.clamp(priorities, min=self.per_epsilon)
        
        # Calculate probabilities
        probs = priorities ** self.alpha
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
        priorities = torch.abs(td_errors) + self.per_epsilon
        priorities = torch.clamp(priorities, min=max(self.per_epsilon, 0.001), max=1e6)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self):
        return self.size


class Actor(nn.Module):
    """Actor network for continuous action space"""
    
    def __init__(self, config: SACConfig):
        super(Actor, self).__init__()
        self.config = config
        
        # Feature embedding for stock data
        self.feature_embedding = nn.Linear(config.num_stock_features, 128)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 128) * 0.02)
        
        # Multi-scale CNN processing
        self.cnn_branches = nn.ModuleList([
            self._create_cnn_branch(128, kernel_size) 
            for kernel_size in [3, 5, 7]
        ])
        
        # Transformer for temporal modeling
        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(128, nhead=8, dropout=0.1)
            for _ in range(2)
        ])
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 128))
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Combined features: multiscale (128*3) + transformer (128) + portfolio (64) = 576
        combined_size = 128 * 3 + 128 + 64
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.actor_hidden_size),
            nn.LayerNorm(config.actor_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(config.actor_hidden_size, config.actor_hidden_size),
            nn.LayerNorm(config.actor_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Action head - outputs mean and log_std for Gaussian policy
        self.action_mean = nn.Linear(config.actor_hidden_size, 1)
        self.action_log_std = nn.Linear(config.actor_hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _create_cnn_branch(self, in_channels: int, kernel_size: int):
        """Create CNN branch for multi-scale processing"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in [self.action_mean, self.action_log_std]:
                    # More aggressive initialization for larger initial actions
                    nn.init.uniform_(module.weight, -0.1, 0.1)  # Increased from 3e-3
                    nn.init.uniform_(module.bias, -0.05, 0.05)  # Increased from 3e-3
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log_std"""
        batch_size = x.size(0)
        
        # Split stock and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # Process stock data
        embedded = self.feature_embedding(stock_data)
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Multi-scale CNN
        multiscale_features = []
        stock_data_cnn = embedded.transpose(1, 2)
        for cnn_branch in self.cnn_branches:
            features = cnn_branch(stock_data_cnn).squeeze(-1)
            multiscale_features.append(features)
        
        # Transformer processing
        transformer_out = embedded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)
        
        # Portfolio processing
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine features
        combined_features = torch.cat([
            *multiscale_features,
            pooled_features,
            portfolio_features
        ], dim=1)
        
        # Shared processing
        shared_out = self.shared_layers(combined_features)
        
        # Action distribution parameters
        mean = self.action_mean(shared_out)
        log_std = self.action_log_std(shared_out)
        log_std = torch.clamp(log_std, min=-10, max=3)  # Increased max from 2 to 3 for higher variance
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sampling
        
        # Apply tanh to bound action to [-1, 1]
        action = torch.tanh(x_t)
        
        # Calculate log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        # Correct for tanh transform: log_prob - log(1 - tanh^2(x))
        log_prob -= torch.log(1 - action.pow(2) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for Q-value estimation"""
    
    def __init__(self, config: SACConfig):
        super(Critic, self).__init__()
        self.config = config
        
        # Feature embedding for stock data
        self.feature_embedding = nn.Linear(config.num_stock_features, 128)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 128) * 0.02)
        
        # Multi-scale CNN processing
        self.cnn_branches = nn.ModuleList([
            self._create_cnn_branch(128, kernel_size) 
            for kernel_size in [3, 5, 7]
        ])
        
        # Transformer for temporal modeling
        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(128, nhead=8, dropout=0.1)
            for _ in range(2)
        ])
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 128))
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Combined features + action: multiscale (128*3) + transformer (128) + portfolio (64) + action (1) = 577
        combined_size = 128 * 3 + 128 + 64 + 1
        
        # Q-value network
        self.q_network = nn.Sequential(
            nn.Linear(combined_size, config.critic_hidden_size),
            nn.LayerNorm(config.critic_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size),
            nn.LayerNorm(config.critic_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.critic_hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _create_cnn_branch(self, in_channels: int, kernel_size: int):
        """Create CNN branch for multi-scale processing"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with state and action"""
        batch_size = x.size(0)
        
        # Split stock and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # Process stock data (same as actor)
        embedded = self.feature_embedding(stock_data)
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Multi-scale CNN
        multiscale_features = []
        stock_data_cnn = embedded.transpose(1, 2)
        for cnn_branch in self.cnn_branches:
            features = cnn_branch(stock_data_cnn).squeeze(-1)
            multiscale_features.append(features)
        
        # Transformer processing
        transformer_out = embedded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)
        
        # Portfolio processing
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine features with action
        combined_features = torch.cat([
            *multiscale_features,
            pooled_features,
            portfolio_features,
            action
        ], dim=1)
        
        # Q-value estimation
        q_value = self.q_network(combined_features)
        
        return q_value


class TradingMode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class SACTradingEnvironment:
    """Trading environment adapted for continuous SAC actions"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 scaled_data: pd.DataFrame, 
                 config: SACConfig, 
                 mode: TradingMode = TradingMode.TRAIN, 
                 device: torch.device = DEVICE,
                 minutes_per_day: int = MINUTES_PER_TRADING_DAY):
        self.data = data
        self.scaled_data = scaled_data
        self.config = config
        self.mode = mode
        self.device = device
        self.minutes_per_day = minutes_per_day
        
        # Calculate daily episode boundaries
        self.total_days = len(data) // self.minutes_per_day
        self.episode_length = self.minutes_per_day
        self.current_day = 0
        
        # Portfolio state normalization
        self.portfolio_normalizer = None
        if config.use_portfolio_normalization:
            self.portfolio_normalizer = PortfolioStateNormalizer(
                warmup_episodes=config.portfolio_warmup_episodes,
                update_frequency=config.portfolio_update_frequency
            )
        
        print(f"SACTradingEnvironment initialized:")
        print(f"  Minutes per day: {self.minutes_per_day}")
        print(f"  Total trading days: {self.total_days}")
        print(f"  Data coverage: {len(data)} minutes")
        
        self.reset()
        
    def reset(self, day_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment for new episode"""
        self.balance = self.config.initial_balance
        self.position = 0.0  # Number of shares
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        # Enhanced tracking
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.position_entry_step = -1
        self.unrealized_pnl = 0.0
        
        # Portfolio tracking for rewards
        self.last_portfolio_value = self.config.initial_balance
        
        # Select day
        if day_idx is not None:
            self.current_day = day_idx
        elif self.mode == TradingMode.TRAIN:
            max_day = max(0, self.total_days - 1)
            self.current_day = np.random.randint(0, max_day + 1) if max_day >= 0 else 0
        else:
            self.current_day = getattr(self, 'last_day', 0)
            self.last_day = (self.current_day + 1) % max(1, self.total_days)
        
        # Set episode boundaries
        self.episode_start = self.current_day * self.minutes_per_day
        self.episode_end = min(self.episode_start + self.episode_length, len(self.data))
        
        # Start with enough data for window
        min_start = max(self.episode_start, self.config.window_size - 1)
        if min_start >= self.episode_end:
            raise ValueError(f"Episode {self.current_day} too short for window size")
            
        self.current_step = min_start
        self.episode_steps_remaining = self.episode_end - self.current_step
        
        # Initialize episode portfolio states for warmup
        self.episode_portfolio_states = []
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state with enhanced features"""
        # Get historical stock data
        start_idx = max(0, self.current_step - self.config.window_size + 1)
        end_idx = self.current_step + 1
        
        # Handle padding if needed
        if start_idx < self.episode_start - self.config.window_size + 1:
            padding_needed = (self.episode_start - self.config.window_size + 1) - start_idx
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
            if padding_needed > 0:
                first_row = stock_data[0:1]
                padding = np.repeat(first_row, padding_needed, axis=0)
                stock_data = np.vstack([padding, stock_data])
        else:
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
        
        # Ensure correct window size
        if len(stock_data) < self.config.window_size:
            padding_needed = self.config.window_size - len(stock_data)
            first_row = stock_data[0:1] if len(stock_data) > 0 else self.scaled_data.iloc[0:1].values
            padding = np.repeat(first_row, padding_needed, axis=0)
            stock_data = np.vstack([padding, stock_data])
        elif len(stock_data) > self.config.window_size:
            stock_data = stock_data[-self.config.window_size:]
        
        # Current price and portfolio calculations
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + (self.position * current_price)
        
        # Update unrealized P&L
        if self.position > 0:
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            market_value = self.position * current_price
            self.unrealized_pnl = (market_value - cost_basis) / cost_basis
        else:
            self.unrealized_pnl = 0.0
        
        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        # Time features
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        time_of_day_normalized = minutes_into_day / self.minutes_per_day
        
        # Session features
        morning_session = 1.0 if minutes_into_day < 120 else 0.0
        midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0
        afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0
        
        # Position timing
        position_holding_time = (self.current_step - self.position_entry_step) if self.position_entry_step >= 0 else 0
        normalized_holding_time = min(position_holding_time / 60, 1.0)
        
        # Position ratio
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0
        
        # Action validity
        can_buy = 1.0 if self.balance > 0 else 0.0
        can_sell = 1.0 if self.position > 0 else 0.0
        
        # Invalid action rate
        recent_invalid_rate = self.invalid_actions / max(self.current_step - self.episode_start + 1, 1)
        normalized_invalid_rate = min(recent_invalid_rate, 1.0)
        
        # Portfolio features
        if self.config.use_portfolio_normalization:
            # Use raw portfolio features - let the normalizer handle scaling
            portfolio_features = [
                self.balance,                     # 0: Raw balance
                self.position * current_price,    # 1: Raw position value  
                portfolio_value,                  # 2: Raw portfolio value
                position_ratio,                   # 3: Position ratio (already 0-1)
                self.unrealized_pnl,              # 4: Raw unrealized P&L
                position_holding_time,            # 5: Raw holding time in steps
                time_of_day_normalized,           # 6: Time of day (already 0-1)
                morning_session,                  # 7: Session indicator (0 or 1)
                midday_session,                   # 8: Session indicator (0 or 1)
                afternoon_session,                # 9: Session indicator (0 or 1)
                can_buy,                          # 10: Can buy flag (0 or 1)
                can_sell,                         # 11: Can sell flag (0 or 1)
                self.invalid_actions              # 12: Raw invalid action count
            ]
        else:
            # Manual normalization (fallback when portfolio normalizer is disabled)
            initial_balance = self.config.initial_balance
            normalized_balance = self.balance / initial_balance if initial_balance > 0 else 0
            normalized_position = self.position * current_price / initial_balance if initial_balance > 0 else 0
            normalized_portfolio_value = portfolio_value / initial_balance if initial_balance > 0 else 0
            
            portfolio_features = [
                normalized_balance,               # 0: Manually normalized balance
                normalized_position,              # 1: Manually normalized position
                normalized_portfolio_value,       # 2: Manually normalized portfolio value
                position_ratio,                   # 3: Position ratio (already 0-1)
                self.unrealized_pnl,              # 4: Raw unrealized P&L
                normalized_holding_time,          # 5: Manually normalized holding time
                time_of_day_normalized,           # 6: Time of day (already 0-1)
                morning_session,                  # 7: Session indicator (0 or 1)
                midday_session,                   # 8: Session indicator (0 or 1) 
                afternoon_session,                # 9: Session indicator (0 or 1)
                can_buy,                          # 10: Can buy flag (0 or 1)
                can_sell,                         # 11: Can sell flag (0 or 1)
                normalized_invalid_rate           # 12: Manually normalized invalid rate
            ]
        
        # Replace non-finite values
        portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
        portfolio_state = np.array(portfolio_features, dtype=np.float32)
        
        # Collect for warmup if needed
        if (self.portfolio_normalizer is not None and not self.portfolio_normalizer.is_fitted):
            self.episode_portfolio_states.append(portfolio_state.copy())
        
        # Apply normalization if fitted
        if (self.portfolio_normalizer is not None and self.portfolio_normalizer.is_fitted):
            portfolio_state = self.portfolio_normalizer.normalize_state(portfolio_state)
        
        # Convert to tensors
        portfolio_state = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device)
        stock_data_state = torch.tensor(stock_data.astype(np.float32), dtype=torch.float32, device=self.device)
        
        # Repeat portfolio state for each timestep and concatenate
        portfolio_state_repeated = portfolio_state.unsqueeze(0).repeat(self.config.window_size, 1)
        combined_state = torch.cat([stock_data_state, portfolio_state_repeated], dim=1)
        
        return combined_state
    
    def _execute_continuous_action(self, action_value: float) -> Tuple[bool, bool]:
        """Execute continuous action and return (trade_executed, invalid_action)"""
        current_price = self.data.iloc[self.current_step]['close']
        trade_executed = False
        invalid_action = False
        
        # Action interpretation:
        # action_value in [-1, 1]
        # Negative: sell proportion of holdings
        # Positive: buy with proportion of available cash
        # Near zero: hold
        
        # TEMPORARILY REMOVED: Apply minimum trade threshold for more active learning
        # if abs(action_value) < self.config.min_trade_amount:
        #     # Hold action - always valid
        #     return False, False
        
        # Only skip truly zero actions (for exact zero from tanh saturation)
        if abs(action_value) < 1e-6:
            return False, False
        
        if action_value > 0:  # Buy action
            if self.balance <= 0:
                # Invalid: no cash available
                invalid_action = True
            else:
                # Calculate buy amount
                cash_to_use = self.balance * action_value * self.config.max_position_size
                shares_to_buy = cash_to_use / current_price
                total_cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                if total_cost <= self.balance:
                    # Execute buy
                    if self.position > 0:
                        # Adding to existing position - calculate weighted average entry price
                        old_value = self.position * self.entry_price
                        new_value = shares_to_buy * current_price
                        total_shares = self.position + shares_to_buy
                        self.entry_price = (old_value + new_value) / total_shares
                        self.position = total_shares
                    else:
                        # New position
                        self.position = shares_to_buy
                        self.entry_price = current_price
                        self.position_entry_step = self.current_step
                    
                    self.balance -= total_cost
                    trade_executed = True
                else:
                    invalid_action = True
                    
        else:  # Sell action (action_value < 0)
            if self.position <= 0:
                # Invalid: no position to sell
                invalid_action = True
            else:
                # Calculate sell amount
                sell_proportion = abs(action_value)
                shares_to_sell = self.position * sell_proportion
                revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_percent)
                
                # Execute sell
                self.balance += revenue
                
                # Calculate profit for this partial sale
                if shares_to_sell > 0:
                    cost_basis = shares_to_sell * self.entry_price * (1 + self.config.transaction_fee_percent)
                    profit = revenue - cost_basis
                    
                    if profit > 0:
                        self.total_profit += profit
                        if sell_proportion == 1.0:  # Full sale
                            self.winning_trades += 1
                    else:
                        self.total_loss += abs(profit)
                        if sell_proportion == 1.0:  # Full sale
                            self.losing_trades += 1
                
                # Update position
                self.position -= shares_to_sell
                
                if self.position < 1e-6:  # Essentially zero
                    self.position = 0.0
                    self.position_entry_step = -1
                    self.total_trades += 1
                
                trade_executed = True
        
        return trade_executed, invalid_action
    
    def step(self, action: float) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute continuous action and return next state, reward, done, info"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Store current portfolio value for comparison
        if not hasattr(self, 'last_portfolio_value'):
            self.last_portfolio_value = self.balance + (self.position * current_price)
        
        current_portfolio_value = self.balance + (self.position * current_price)
        
        # Execute action
        trade_executed, invalid_action = self._execute_continuous_action(action)
        
        # Track invalid actions
        if invalid_action:
            self.invalid_actions += 1
        
        # Calculate reward (same system as DQN v5)
        new_portfolio_value = self.balance + (self.position * current_price)
        portfolio_change = new_portfolio_value - self.last_portfolio_value
        
        # Portfolio value change reward
        portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.1)
        reward = portfolio_change * portfolio_scaling
        
        # Invalid action penalty
        if invalid_action:
            invalid_penalty = getattr(self.config, 'invalid_penalty', 0.1)
            reward -= invalid_penalty
        
        # Update portfolio value for next step
        self.last_portfolio_value = new_portfolio_value
        
        # Move to next step
        self.current_step += 1
        self.episode_steps_remaining -= 1
        
        # Check if episode is done
        done = (self.current_step >= self.episode_end or 
                self.episode_steps_remaining <= 0 or 
                self.balance <= 0)
        
        # Force close positions at end of day
        if done and self.position > 0:
            revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0.0
            self.total_trades += 1
            
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
            else:
                self.losing_trades += 1
                self.total_loss += abs(profit)
            
            # Final reward calculation
            final_portfolio_value = self.balance
            final_change = final_portfolio_value - self.last_portfolio_value
            reward += final_change * portfolio_scaling
        
        # Get next state
        if done:
            self.current_step -= 1
            next_state = self._get_state()
            self.current_step += 1
        else:
            next_state = self._get_state()
        
        # Info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'invalid_action': invalid_action,
            'invalid_actions': self.invalid_actions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'current_day': self.current_day,
            'episode_steps_remaining': self.episode_steps_remaining,
            'unrealized_pnl': self.unrealized_pnl,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'action_value': action,
            'final_reward': reward
        }
        
        return next_state, reward, done, info


class SAC:
    """Soft Actor-Critic agent for continuous action trading"""
    
    def __init__(self, config: SACConfig, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        
        # Initialize networks
        self.actor = Actor(config).to(device)
        self.critic1 = Critic(config).to(device)
        self.critic2 = Critic(config).to(device)
        
        # Target critics for stability
        self.target_critic1 = Critic(config).to(device)
        self.target_critic2 = Critic(config).to(device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=config.critic_learning_rate)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=config.critic_learning_rate)
        
        # Automatic entropy tuning
        if config.alpha_auto_tune:
            self.target_entropy = config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=config.alpha_learning_rate)
        else:
            self.alpha = config.initial_alpha
        
        # Replay buffer
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config, device)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
        
        # Print network info
        total_params = sum(p.numel() for p in self.actor.parameters()) + \
                      sum(p.numel() for p in self.critic1.parameters()) + \
                      sum(p.numel() for p in self.critic2.parameters())
        print(f"SAC networks initialized: {total_params:,} total parameters")
    
    @property
    def alpha(self):
        """Get current alpha value"""
        if self.config.alpha_auto_tune:
            return self.log_alpha.exp()
        else:
            return self._alpha
    
    @alpha.setter 
    def alpha(self, value):
        """Set alpha value (only when not auto-tuning)"""
        if not self.config.alpha_auto_tune:
            self._alpha = value
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> float:
        """Select action from policy"""
        self.actor.eval()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            if deterministic:
                # Use mean action for evaluation
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
            else:
                # Sample from policy
                action, _ = self.actor.sample(state)
            
            action = action.cpu().item()
        self.actor.train()
        return action
    
    def update(self) -> Dict[str, float]:
        """Perform SAC update with prioritized experience replay"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # Calculate current beta for importance sampling
        beta = self.config.per_beta_start + (self.config.per_beta_end - self.config.per_beta_start) * \
               min(1.0, self.steps_done / 100000)  # Beta annealing over 100k steps
        
        try:
            # Sample batch with prioritized sampling
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.config.batch_size, beta)
            
            # Update critics
            with torch.no_grad():
                # Sample actions for next states
                next_actions, next_log_probs = self.actor.sample(next_states)
                
                # Target Q-values using minimum of two critics
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                
                target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1).float()) * self.config.gamma * target_q
            
            # Current Q-values
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            # Calculate TD errors for priority updates
            td_errors_1 = (current_q1 - target_q).squeeze()
            td_errors_2 = (current_q2 - target_q).squeeze()
            
            # Clamp TD errors to prevent extreme values
            td_errors_1 = torch.clamp(td_errors_1, min=-10.0, max=10.0)
            td_errors_2 = torch.clamp(td_errors_2, min=-10.0, max=10.0)
            
            # Replace any NaN/inf values with zero
            td_errors_1 = torch.where(torch.isfinite(td_errors_1), td_errors_1, torch.zeros_like(td_errors_1))
            td_errors_2 = torch.where(torch.isfinite(td_errors_2), td_errors_2, torch.zeros_like(td_errors_2))
            
            # Combined TD error for priority update (average of both critics)
            td_errors = (torch.abs(td_errors_1) + torch.abs(td_errors_2)) / 2.0
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors.detach())
            
            # Weighted critic losses using importance sampling weights
            critic1_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none').squeeze()).mean()
            critic2_loss = (weights * F.mse_loss(current_q2, target_q, reduction='none').squeeze()).mean()
            
            # Update critics
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_optimizer.step()
            
            # Update actor
            new_actions, log_probs = self.actor.sample(states)
            q1_new = self.critic1(states, new_actions)
            q2_new = self.critic2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Weighted actor loss using importance sampling weights
            actor_loss = (weights * (self.alpha * log_probs - q_new).squeeze()).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Update alpha (if auto-tuning)
            alpha_loss = 0
            if self.config.alpha_auto_tune:
                # Weighted alpha loss using importance sampling weights
                alpha_loss = -(weights * (self.log_alpha * (log_probs + self.target_entropy).detach()).squeeze()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
            
            return {
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item() if self.config.alpha_auto_tune else 0,
                'alpha': self.alpha.item(),
                'mean_q1': current_q1.mean().item(),
                'mean_q2': current_q2.mean().item(),
                'beta': beta,
                'mean_td_error': td_errors.mean().item()
            }
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA Error in SAC update: {e}")
                print(f"Memory size: {len(self.memory)}")
                print(f"Steps done: {self.steps_done}")
                # Try to recover by clearing CUDA cache
                torch.cuda.empty_cache()
                return {}
            else:
                raise
    
    def _soft_update(self, target, source):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def train_episode(self, env: SACTradingEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Track update metrics
        update_metrics = {
            'critic1_loss': [],
            'critic2_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha': [],
            'mean_q1': [],
            'mean_q2': []
        }
        
        while True:
            # Select action
            action = self.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, done)
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            self.steps_done += 1
            
            # Update every step (if enough samples)
            if self.steps_done % self.config.update_frequency == 0 and len(self.memory) >= self.config.batch_size:
                update_info = self.update()
                # Track metrics
                for key, value in update_info.items():
                    if key in update_metrics:
                        update_metrics[key].append(value)
            
            state = next_state
            
            if done:
                break
        
        self.episodes_done += 1
        
        # Calculate return
        final_value = info['balance'] + (info['position'] * info['current_price'])
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        
        # Average update metrics
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
            'winning_trades': info['winning_trades'],
            'losing_trades': info['losing_trades'],
            'invalid_actions': info['invalid_actions'],
            **avg_update_metrics
        }
    
    def save(self, path: str, portfolio_normalizer=None):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.config.alpha_auto_tune else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.config.alpha_auto_tune else None,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'config': self.config
        }, path)
        
        # Save normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.save(normalizer_path)
    
    def load(self, path: str, portfolio_normalizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.config.alpha_auto_tune and checkpoint.get('log_alpha') is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        
        # Load normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.load(normalizer_path)


def filter_to_regular_hours(df):
    """Filter dataframe to regular market hours using UTC timestamps"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Regular market hours filtering
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    eastern_times = df['timestamp'].dt.tz_convert('US/Eastern')
    market_open = eastern_times.dt.time >= time(9, 30)
    market_close = eastern_times.dt.time < time(16, 0)
    
    filtered_df = df[market_open & market_close].reset_index(drop=True)
    filtered_df['timestamp'] = filtered_df['timestamp'].dt.tz_localize(None)
    
    print(f"Data filtered: {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df


def load_stock_data(data_path: str, cutoff: pd.Timestamp | None = None, cols_to_keep: list[str] = STOCK_FEATURES_V2) -> tuple[pd.DataFrame, datetime, datetime]:
    """Get saved csv data and filter to regular market hours"""
    df = pd.read_csv(data_path)

    # Apply cutoff first if specified
    if cutoff:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if cutoff.tz is not None:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(cutoff.tz)
        else:
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        df = df[df['timestamp'] >= cutoff]
        
    # Filter to regular market hours
    df = filter_to_regular_hours(df)
    
    # Get date range after filtering
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
        
    return df[cols_to_keep], start_date, end_date


def train_sac(data_path: str,
              cutoff: pd.Timestamp,
              num_episodes: int = NUM_EPISODES,
              save_interval: int = 100,
              validation_frequency: int = EVALUATE_INTERVAL,
              early_stopping_patience: int = 15,
              train_ratio: float = TRAIN_RATIO,
              valid_ratio: float = VALID_RATIO,
              use_preprocessing: bool = True,
              scaling_method: str = 'robust',
              outlier_method: str = 'winsorize',
              preprocessor_save_path: Optional[str] = None) -> Dict:
    """
    Main SAC training function with validation and early stopping
    
    Args:
        data_path: Path to CSV file with stock data
        cutoff: Timestamp to start data from
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        validation_frequency: Run validation every N episodes
        early_stopping_patience: Stop if validation doesn't improve for N checks
        train_ratio: Ratio of data for training
        valid_ratio: Ratio of data for validation
        use_preprocessing: Whether to apply preprocessing
        scaling_method: Method for scaling features
        outlier_method: Method for handling outliers
        preprocessor_save_path: Path to save the fitted preprocessor
    """
    
    # Load data
    print("Loading data for SAC training...")
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
        from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
        
        if preprocessor_save_path is None:
            from pathlib import Path
            data_name = Path(data_path).stem
            preprocessor_save_path = f"sac_preprocessor_{data_name}.pkl"
        
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
        train_data_scaled = train_data
        valid_data_scaled = valid_data
        test_data_scaled = test_data
    
    # Initialize configuration and environments
    config = SACConfig()
    
    train_env = SACTradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = SACTradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = SACTradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    print(f"train_data_scaled.shape: {train_data_scaled.shape}")
    print(f"valid_data_scaled.shape: {valid_data_scaled.shape}")
    print(f"test_data_scaled.shape: {test_data_scaled.shape}")
    
    # Create SAC agent
    agent = SAC(config)
    
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
    
    print("Starting SAC training...")
    
    # Phase 1: Portfolio State Warmup (if needed)
    if train_env.portfolio_normalizer is not None and not train_env.portfolio_normalizer.is_fitted:
        warmup_episodes = train_env.portfolio_normalizer.warmup_episodes
        print(f"\n{'='*60}")
        print(f"PORTFOLIO NORMALIZATION WARMUP PHASE")
        print(f"{'='*60}")
        print(f"Collecting portfolio states for {warmup_episodes} episodes...")
        print("(No training will occur during this phase)")
        
        for warmup_ep in range(warmup_episodes):
            state = train_env.reset()
            episode_portfolio_states = []
            
            while True:
                # Random action during warmup
                action = np.random.uniform(-1, 1)
                next_state, reward, done, info = train_env.step(action)
                
                # Collect portfolio states
                if hasattr(train_env, 'episode_portfolio_states'):
                    episode_portfolio_states.extend(train_env.episode_portfolio_states)
                
                state = next_state
                if done:
                    break
            
            # Add collected states to normalizer
            if episode_portfolio_states:
                train_env.portfolio_normalizer.collect_warmup_data(episode_portfolio_states)
            train_env.portfolio_normalizer.increment_episode()
            
            # Progress update
            if (warmup_ep + 1) % 10 == 0 or warmup_ep == warmup_episodes - 1:
                progress = (warmup_ep + 1) / warmup_episodes
                print(f"  Warmup progress: {warmup_ep + 1}/{warmup_episodes} ({progress*100:.1f}%)")
        
        print(f"\n[PORTFOLIO COLLECTION COMPLETE]")
        print(f"[FITTING NORMALIZER] Fitting portfolio normalizer...")
        print(f"[READY] Ready to start SAC training with normalized portfolio features!")
        print(f"\n{'='*60}")
        print(f"SAC TRAINING PHASE")
        print(f"{'='*60}")
    
    # Main training loop
    for episode in range(num_episodes):
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
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Winning/Losing: {metrics['winning_trades']}/{metrics['losing_trades']}")
        print(f"  Invalid Actions: {metrics['invalid_actions']}")
        print(f"  Steps: {metrics['episode_steps']}")
        
        # Print SAC-specific metrics
        if 'alpha' in metrics:
            print(f"  Alpha: {metrics['alpha']:.4f}")
        if 'actor_loss' in metrics:
            print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
        if 'critic1_loss' in metrics:
            print(f"  Critic Losses: {metrics['critic1_loss']:.4f} / {metrics['critic2_loss']:.4f}")
        
        # Portfolio normalizer status
        if train_env.portfolio_normalizer is not None:
            print(f"  Portfolio Normalizer: [ACTIVE]")
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print("\nRunning SAC validation...")
            
            val_state = val_env.reset()
            val_reward = 0
            val_done = False
            
            while not val_done:
                val_action = agent.select_action(val_state, deterministic=True)  # Deterministic for validation
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
            print(f"  Winning/Losing: {val_info['winning_trades']}/{val_info['losing_trades']}")
            print(f"  Invalid Actions: {val_info['invalid_actions']}")
            
            # Early stopping check
            if val_return > best_validation_return:
                best_validation_return = val_return
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    'actor': agent.actor.state_dict(),
                    'critic1': agent.critic1.state_dict(),
                    'critic2': agent.critic2.state_dict(),
                    'target_critic1': agent.target_critic1.state_dict(),
                    'target_critic2': agent.target_critic2.state_dict(),
                    'steps_done': agent.steps_done,
                    'episodes_done': agent.episodes_done
                }
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at episode {episode+1}")
                print(f"Best validation return: {best_validation_return:.2%}")
                
                # Restore best model
                if best_model_state:
                    agent.actor.load_state_dict(best_model_state['actor'])
                    agent.critic1.load_state_dict(best_model_state['critic1'])
                    agent.critic2.load_state_dict(best_model_state['critic2'])
                    agent.target_critic1.load_state_dict(best_model_state['target_critic1'])
                    agent.target_critic2.load_state_dict(best_model_state['target_critic2'])
                    agent.steps_done = best_model_state['steps_done']
                    agent.episodes_done = best_model_state['episodes_done']
                break
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"sac_checkpoint_episode_{episode}.pt"
            agent.save(checkpoint_path, train_env.portfolio_normalizer)
            print(f"Saved SAC checkpoint at episode {episode}")
    
    # Multi-day test evaluation
    print("\n" + "="*50)
    print("SAC MULTI-DAY TEST EVALUATION (6 DAYS)")
    print("="*50)
    
    num_test_days = min(6, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning SAC backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_portfolio_values = [config.initial_balance]
        test_price_history = []
        
        while not test_done:
            test_action = agent.select_action(test_state, deterministic=True)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            # Track for plotting
            test_action_history.append(test_action)
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
        
        # Calculate metrics
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - config.initial_balance) / config.initial_balance
        
        # Performance metrics
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            sharpe = test_return / portfolio_volatility if portfolio_volatility > 1e-8 else test_return * 10
        else:
            sharpe = 0.0
        
        day_results = {
            'day_idx': day_idx,
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'invalid_actions': test_info['invalid_actions'],
            'episode_reward': test_reward
        }
        
        all_test_results.append(day_results)
        all_portfolio_values.append(test_portfolio_values)
        all_price_histories.append(test_price_history)
        all_action_histories.append(test_action_history)
        
        print(f"  Day {day_idx + 1} Results:")
        print(f"    Final Value: ${test_final_value:,.2f}")
        print(f"    Return: {test_return:.2%}")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        print(f"    Trades: {test_info['total_trades']} (W:{test_info['winning_trades']}, L:{test_info['losing_trades']})")
        print(f"    Invalid Actions: {test_info['invalid_actions']}")
    
    # Calculate aggregate statistics
    returns = [r['total_return'] for r in all_test_results]
    final_values = [r['final_value'] for r in all_test_results]
    
    print(f"\n{'='*50}")
    print("AGGREGATE SAC TEST RESULTS")
    print(f"{'='*50}")
    print(f"Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
    print(f"Best Return: {np.max(returns):.2%}")
    print(f"Worst Return: {np.min(returns):.2%}")
    print(f"Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    print(f"Average Final Value: ${np.mean(final_values):,.2f}")
    
    # Create results dictionary
    results = {
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
        'multi_day_test_results': {
            'individual_days': all_test_results,
            'portfolio_values': all_portfolio_values,
            'price_histories': all_price_histories,
            'action_histories': all_action_histories,
            'aggregate_stats': {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'win_rate': np.sum([r > 0 for r in returns]) / len(returns),
                'avg_final_value': np.mean(final_values)
            }
        },
        'start_date': start_date,
        'end_date': end_date
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    data_path = "data/TSLA_1min_features.csv"
    cutoff = pd.Timestamp("2023-01-01")
    
    results = train_sac(
        data_path=data_path,
        cutoff=cutoff,
        num_episodes=200,
        save_interval=50,
        validation_frequency=20
    )
    
    print("SAC training completed!")
    print(f"Final results: {results['multi_day_test_results']['aggregate_stats']}")
