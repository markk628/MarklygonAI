"""
DQN v5: Enhanced Trading Agent with Portfolio Tracking + Invalid Action Learning
===============================================================================

REWARD SYSTEM FIX APPLIED:
- Removed auxiliary rewards and inconsistent scaling
- Implemented unified portfolio value change tracking for ALL actions  
- ALL ACTIONS: Portfolio value change * 0.1 (consistent scaling)
- INVALID ACTIONS: Additional -0.1 penalty to teach action validity

This maintains reward-return alignment through portfolio tracking while
ensuring the agent learns which actions are valid.
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
    DEVICE,
    EPSILON_EARLY_STOPPING_THRESHOLD,
    STOCK_FEATURES_V2,
    NUM_EPISODES,
    TRAIN_RATIO,
    VALID_RATIO,
    EVALUATE_INTERVAL,
    MINUTES_PER_TRADING_DAY
)

from src.models.mark.dqn_v2.config import TradingConfig, ArchitectureType
from src.models.mark.dqn_v2.normalization import PortfolioStateNormalizer
from src.models.mark.dqn_v2.networks import create_network


class PrioritizedReplayBufferGPU:
    """PER VRAM version"""
    
    def __init__(self, capacity: int, config: TradingConfig, device: torch.device=DEVICE):
        self.capacity = capacity
        self.config = config
        self.device = device
        self.position = 0
        self.size = 0

        self.states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01
        self.max_priority = 1.0
        
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool):
        """save experience"""
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
        probs = priorities ** self.config.alpha
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
                 device: torch.device=DEVICE,
                 minutes_per_day: int = MINUTES_PER_TRADING_DAY):
        self.data = data
        self.scaled_data = scaled_data
        self.config = config
        self.mode = mode
        self.device = device
        
        # Set minutes per day based on parameter or use regular market hours as default
        self.minutes_per_day = minutes_per_day
        
        # Calculate daily episode boundaries
        self.total_days = len(data) // self.minutes_per_day
        self.episode_length = self.minutes_per_day
        self.current_day = 0
        
        # Validate data structure
        expected_total_minutes = self.total_days * self.minutes_per_day
        actual_minutes = len(data)
        unused_minutes = actual_minutes - expected_total_minutes
        
        if unused_minutes > 0:
            print(f"Warning: {unused_minutes} minutes of data will be unused due to incomplete trading days")
        
        # Ensure we have enough data for at least one complete episode with window
        min_required = self.minutes_per_day + config.window_size
        if actual_minutes < min_required:
            raise ValueError(f"Insufficient data: need at least {min_required} minutes, got {actual_minutes}")
        
        print(f"TradingEnvironment initialized:")
        print(f"  Minutes per day: {self.minutes_per_day}")
        print(f"  Total trading days: {self.total_days}")
        print(f"  Data coverage: {expected_total_minutes} / {actual_minutes} minutes")
        print(f"  Data utilization: {expected_total_minutes/actual_minutes*100:.1f}%")
        
        # Initialize tracking variables
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.position_entry_step = -1
        self.consecutive_invalid_actions = 0
        self.last_action = 0
        self.unrealized_pnl = 0.0
        
        # Portfolio state normalization
        self.portfolio_normalizer = None
        if config.use_portfolio_normalization:
            self.portfolio_normalizer = PortfolioStateNormalizer(
                warmup_episodes=config.portfolio_warmup_episodes,
                update_frequency=config.portfolio_update_frequency
            )
        
        self.reset()
        
    def reset(self, day_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment to initial state for a new trading day"""
        self.balance = self.config.initial_balance
        self.position = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        # Reset enhanced tracking variables
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.position_entry_step = -1  # Track when position was entered
        self.consecutive_invalid_actions = 0
        self.consecutive_holds = 0  # Track consecutive hold actions for patience bonus
        self.last_action = 0  # Track last action (0=hold, 1=buy, 2=sell)
        self.unrealized_pnl = 0.0
        
        # Enhanced trading discipline tracking
        self.last_trade_step = -100  # When last buy/sell occurred (start far back)
        self.last_trade_was_loss = False  # Whether last completed trade was a loss
        self.steps_since_last_loss = 0  # Steps since last losing trade
        
        # Initialize portfolio tracking for rewards
        self.last_portfolio_value = self.config.initial_balance
        
        # Select which day to trade
        if day_idx is not None:
            self.current_day = day_idx
        elif self.mode == TradingMode.TRAIN:
            # Random day for training to ensure good exploration
            max_day = max(0, self.total_days - 1)
            self.current_day = np.random.randint(0, max_day + 1) if max_day >= 0 else 0
        else:
            # Sequential days for validation/testing
            self.current_day = getattr(self, 'last_day', 0)
            self.last_day = (self.current_day + 1) % max(1, self.total_days)
        
        # Set episode boundaries
        self.episode_start = self.current_day * self.minutes_per_day
        self.episode_end = min(self.episode_start + self.episode_length, len(self.data))
        
        # Ensure there is enough data for the window
        # Start the episode far enough in to have a full window
        min_start = max(self.episode_start, self.config.window_size - 1)
        
        # Ensure we don't start too late in the episode
        if min_start >= self.episode_end:
            raise ValueError(f"Episode {self.current_day} too short for window size {self.config.window_size}")
            
        self.current_step = min_start
        self.episode_steps_remaining = self.episode_end - self.current_step
        
        # Validate episode has minimum required steps
        if self.episode_steps_remaining < 10:  # Minimum reasonable episode length
            print(f"Warning: Very short episode {self.current_day}: only {self.episode_steps_remaining} steps")
        
        # Initialize episode portfolio state collection (used during warmup phase)
        self.episode_portfolio_states = []
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state with enhanced features for intraday trading"""
        # Get historical data
        start_idx = max(0, self.current_step - self.config.window_size + 1)
        end_idx = self.current_step + 1
        
        # If not enough historical data, pad with the earliest available data
        if start_idx < self.episode_start - self.config.window_size + 1:
            # Pad with the first available data point in the episode
            padding_needed = (self.episode_start - self.config.window_size + 1) - start_idx
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
            if padding_needed > 0:
                first_row = stock_data[0:1]  # Get first row
                padding = np.repeat(first_row, padding_needed, axis=0)
                stock_data = np.vstack([padding, stock_data])
        else:
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
        
        # Ensure there is exactly window_size rows
        if len(stock_data) < self.config.window_size:
            padding_needed = self.config.window_size - len(stock_data)
            first_row = stock_data[0:1] if len(stock_data) > 0 else self.scaled_data.iloc[0:1].values
            padding = np.repeat(first_row, padding_needed, axis=0)
            stock_data = np.vstack([padding, stock_data])
        elif len(stock_data) > self.config.window_size:
            stock_data = stock_data[-self.config.window_size:]
        
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + (self.position * current_price)
        
        # Update unrealized P&L if holding position
        if self.position > 0:
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            market_value = self.position * current_price
            self.unrealized_pnl = (market_value - cost_basis) / cost_basis
        else:
            self.unrealized_pnl = 0.0
        
        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        # Calculate intraday time features (regular market hours: 9:30 AM - 4:00 PM = 390 minutes)
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        time_of_day_normalized = minutes_into_day / self.minutes_per_day  # 0 to 1
        
        # Market session features
        morning_session = 1.0 if minutes_into_day < 120 else 0.0  # (9:30-11:30)
        midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0  # (11:30-2:00)
        afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0  # (2:00-4:00)
        
        # Position timing features
        position_holding_time = (self.current_step - self.position_entry_step) if self.position_entry_step >= 0 else 0
        normalized_holding_time = min(position_holding_time / 60, 1.0)  # Normalize to 1 hour max
        
        # Enhanced portfolio features for intraday trading
        initial_balance = self.config.initial_balance
        normalized_balance = self.balance / initial_balance if initial_balance > 0 else 0
        normalized_position = self.position * current_price / initial_balance if initial_balance > 0 else 0
        normalized_portfolio_value = portfolio_value / initial_balance if initial_balance > 0 else 0
        
        # Position ratio and risk metrics
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0
        
        # Action validity flags (match execution logic exactly)
        position_value = self.balance * self.config.max_position_size
        shares_to_buy = int(position_value / current_price) if current_price > 0 else 0
        total_cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)

        can_buy = 1.0 if (self.position == 0 and 
                          shares_to_buy > 0 and 
                          total_cost <= self.balance) else 0.0
        can_sell = 1.0 if self.position > 0 else 0.0
        
        # Add invalid action frequency context (help agent learn patterns)
        recent_invalid_rate = self.invalid_actions / max(self.current_step - self.episode_start + 1, 1)
        normalized_invalid_rate = min(recent_invalid_rate, 1.0)  # Cap at 100%
        
        # Portfolio features
        portfolio_features = [
            # Core current metrics (4)
            normalized_balance,           # Current cash available
            normalized_position,          # Current stock holdings
            normalized_portfolio_value,   # Current total value
            position_ratio,               # Current position size ratio
            
            # Current position status (2)
            self.unrealized_pnl,          # Current position P&L
            normalized_holding_time,      # How long holding current position
            
            # Current timing context (4)
            time_of_day_normalized,       # Where in trading day
            morning_session,              # Current market session
            midday_session,
            afternoon_session,
            
            # Current action validity (3)
            can_buy,                      # Can execute buy now
            can_sell,                     # Can execute sell now
            normalized_invalid_rate       # Recent invalid action frequency (helps learn patterns)
        ]
        
        # Replace any non-finite values with 0
        portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
        
        # Convert to numpy array
        portfolio_state = np.array(portfolio_features, dtype=np.float32)
        
        # Collect state for warmup if normalizer exists and is in warmup phase
        if (self.portfolio_normalizer is not None and 
            not self.portfolio_normalizer.is_fitted):
            self.episode_portfolio_states.append(portfolio_state.copy())
        
        # Apply normalization if fitted
        if (self.portfolio_normalizer is not None and 
            self.portfolio_normalizer.is_fitted):
            portfolio_state = self.portfolio_normalizer.normalize_state(portfolio_state)
        
        # Convert to tensor
        portfolio_state = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device)
        
        # Convert stock data to tensor
        stock_data_state = torch.tensor(stock_data.astype(np.float32), dtype=torch.float32, device=self.device)
        
        # Repeat portfolio state for each timestep and concatenate
        # This is needed for compatibility with the current state representation
        # The network will extract portfolio features from the first timestep
        # Takes up more memory, but is faster than reconstructing the flattened stock data
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
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                return cost > self.balance
        elif action == 2:
            return self.position == 0
        return False
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state, reward, done, info
        
        OPTIMIZABLE REWARD SYSTEM - Pure Portfolio Value Change + Invalid Penalty:
        - ALL ACTIONS: Portfolio value change * portfolio_scaling (configurable)
        - INVALID ACTIONS: Additional -invalid_penalty to teach action validity
        
        This maintains portfolio tracking for performance while adding
        direct feedback for invalid actions. Parameters can be optimized
        using the Optuna-based optimization script.
        """
            
        current_price = self.data.iloc[self.current_step]['close']        
        reward = 0
        trade_executed = False
        invalid_action = self._is_invalid_action(action)
        
        # Store current portfolio value for comparison
        if not hasattr(self, 'last_portfolio_value'):
            self.last_portfolio_value = self.balance + (self.position * current_price)
        
        # Calculate current portfolio value
        current_portfolio_value = self.balance + (self.position * current_price)
        
        # PURE PORTFOLIO TRACKING: Reward = portfolio value change (always)
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.010)  # Support optimization
        reward = portfolio_change * portfolio_scaling  # Consistent scaling for all actions
        
        # Update last portfolio value for next step
        self.last_portfolio_value = current_portfolio_value
        
        # Execute the action (portfolio change + invalid penalty if needed)
        if invalid_action:
            self.invalid_actions += 1
            self.consecutive_invalid_actions += 1
            # Add penalty for invalid actions to teach action validity
            invalid_penalty = getattr(self.config, 'invalid_penalty', 0.497)  # Support optimization
            reward -= invalid_penalty  # Configurable penalty for invalid actions
        else:
            self.consecutive_invalid_actions = 0 
            
            if action == 1:  # Buy
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                self.position = shares_to_buy
                self.balance -= cost
                self.entry_price = current_price
                self.position_entry_step = self.current_step
                trade_executed = True
                self.last_action = 1
                self.consecutive_holds = 0  # Reset hold counter
                
                # Update trade tracking
                self.last_trade_step = self.current_step
                self.steps_since_last_loss += 1
                    
            elif action == 2:  # Sell
                revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
                cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
                profit = revenue - cost_basis
                
                self.balance += revenue
                self.position = 0
                trade_executed = True
                self.total_trades += 1
                self.last_action = 2
                self.consecutive_holds = 0  # Reset hold counter
                
                # Update trade tracking
                self.last_trade_step = self.current_step
                self.last_trade_was_loss = profit <= 0
                if profit <= 0:
                    self.steps_since_last_loss = 0  # Reset counter on loss
                else:
                    self.steps_since_last_loss += 1
                
                if profit > 0:
                    self.winning_trades += 1
                    self.total_profit += profit
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(profit)
                
                self.position_entry_step = -1
                
            else:  # Hold (action == 0)
                self.last_action = 0
                self.consecutive_holds += 1
        
        # Move to next step
        self.current_step += 1
        self.episode_steps_remaining -= 1
        
        # Check if episode is done (end of trading day or out of balance)
        done = (self.current_step >= self.episode_end or 
                self.episode_steps_remaining <= 0 or 
                self.balance <= 0)
        
        # Force close any open positions at end of day (realistic intraday trading)
        if done and self.position > 0:
            # Close position at current price
            revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0
            self.total_trades += 1
            
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
            else:
                self.losing_trades += 1
                self.total_loss += abs(profit)
            
            # Update portfolio value after forced close for final reward calculation
            final_portfolio_value = self.balance
            final_change = final_portfolio_value - self.last_portfolio_value
            portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.010)
            reward += final_change * portfolio_scaling  # Same consistent scaling
        
        # Get next state (or terminal state if done)
        if done:
            # Return current state as next state when episode is done
            self.current_step -= 1
            next_state = self._get_state()
            self.current_step += 1
        else:
            # For non-terminal transitions, use the state at the new current_step
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
            'losing_trades': self.losing_trades,
            'current_day': self.current_day,
            'episode_steps_remaining': self.episode_steps_remaining,
            'unrealized_pnl': getattr(self, 'unrealized_pnl', 0.0),
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'consecutive_holds': self.consecutive_holds,
            'steps_since_last_trade': self.current_step - self.last_trade_step,
            'last_trade_was_loss': self.last_trade_was_loss,
            'steps_since_last_loss': self.steps_since_last_loss,
            'final_reward': reward  # Track the final reward for analysis
        }
        
        return next_state, reward, done, info


class DoubleDuelingDQN:
    """Double Dueling DQN Agent with PER and configurable architectures"""
    
    def __init__(self, config: TradingConfig, device: torch.device=DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        print(f"Using architecture: {config.architecture_type.value}")
        
        # Networks using factory function
        self.q_network = create_network(config).to(device)
        self.target_network = create_network(config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Print network size for debugging
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        print(f"Network parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                    lr=config.learning_rate, 
                                    weight_decay=1e-5)  # Fixed weight_decay
        
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
        """Select action using epsilon-greedy policy with minimum profit threshold"""
        if epsilon is None:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if random.random() > epsilon:
            self.q_network.eval()  # Set to evaluation mode for deterministic inference
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                
                # Apply minimum profit threshold: only trade if significantly better than holding
                hold_q_value = q_values[0, 0]  # Q-value for holding (action 0)
                
                # Check if buy/sell actions meet minimum profit threshold
                for action in [1, 2]:  # Buy and Sell
                    if len(q_values[0]) > action:  # Ensure action exists
                        profit_advantage = q_values[0, action] - hold_q_value
                        if profit_advantage < self.config.min_profit_threshold:
                            q_values[0, action] = -float('inf')  # Don't trade unless advantage is clear
                
                action = torch.argmax(q_values, dim=1).item()
            self.q_network.train()  # Set back to training mode
            return action
        else:
            return random.randrange(self.config.num_actions)
    
    def update(self) -> Dict[str, float]:
        """Perform one update step"""
        if len(self.memory) < self.config.batch_size:
            return {}
        

        
        try:
            # Calculate current beta for importance sampling
            beta = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * min(1.0, self.steps_done / 100000)  # Fixed beta decay steps
            
            # Sample batch
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config.batch_size, beta)
            
            # Compute current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Double DQN: use online network to select actions, target network to evaluate
            with torch.no_grad():
                if self.config.use_double_dqn:
                    next_actions = self.q_network(next_states).max(1)[1]
                    next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
                else:
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                    
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
            
            # Soft update of target network using tau
            if self.config.tau > 0:
                for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
            else:
                # Hard update every target_update_frequency steps
                self.update_count += 1
                if self.update_count % self.config.target_update_frequency == 0:
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
            
            # Perform update every update_frequency steps
            if self.steps_done % self.config.update_frequency == 0:
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
            'winning_trades': info['winning_trades'],
            'losing_trades': info['losing_trades'],
            'invalid_actions': info['invalid_actions'],
            **avg_update_metrics
        }
    
    def save(self, path: str, portfolio_normalizer=None):
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
        
        # Save portfolio normalizer separately if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.save(normalizer_path)
    
    def load(self, path: str, portfolio_normalizer=None):
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
        
        # Load portfolio normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.load(normalizer_path)


# TODO: probably not needed
# feature engineering v2 handles this
def filter_to_regular_hours(df):
    """Filter dataframe to regular market hours using UTC timestamps
    
    Regular market hours: 9:30 AM - 4:00 PM EST
    In UTC: 14:30 - 21:00 (EST, winter) or 13:30 - 20:00 (EDT, summer)
    Note: Assumes weekends are already filtered out during feature engineering
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Regular market hours filtering (handles DST automatically through pandas)
    # First localize to UTC if timezone-naive, then convert to Eastern time
    if df['timestamp'].dt.tz is None:
        # Assume naive timestamps are UTC
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    eastern_times = df['timestamp'].dt.tz_convert('US/Eastern')
    market_open = eastern_times.dt.time >= time(9, 30)
    market_close = eastern_times.dt.time < time(16, 0)
    
    # Apply filters and keep original timestamps (convert back to naive for consistency)
    filtered_df = df[market_open & market_close].reset_index(drop=True)
    filtered_df['timestamp'] = filtered_df['timestamp'].dt.tz_localize(None)  # Remove timezone info
    
    print(f"Data filtered: {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df

def load_stock_data(data_path: str, cutoff: pd.Timestamp | None=None, cols_to_keep: list[str]=STOCK_FEATURES_V2) -> tuple[pd.DataFrame, datetime, datetime]:
    """
    Get saved csv data and filter to regular market hours
    """
    df = pd.read_csv(data_path)

    # Apply cutoff first if specified
    if cutoff:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure cutoff and data timestamps are timezone-aware and compatible
        if cutoff.tz is not None:
            # If cutoff has timezone, convert data timestamps to same timezone
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(cutoff.tz)
        else:
            # If cutoff is naive, ensure data timestamps are also naive
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        df = df[df['timestamp'] >= cutoff]
        
    # Filter to regular market hours
    df = filter_to_regular_hours(df)
    
    # Get date range after filtering
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
        
    return df[cols_to_keep], start_date, end_date

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
              preprocessor_save_path: Optional[str] = None,
              architecture_type: ArchitectureType = ArchitectureType.IMPROVED):
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
        architecture_type: Architecture type for the DQN ('original', 'improved', 'hybrid')
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
        from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
        
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
    config = TradingConfig(architecture_type=architecture_type)
    
    # Create environments
    train_env = TradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = TradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = TradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    print(f"train_data_scaled.shape: {train_data_scaled.shape}")
    print(f"valid_data_scaled.shape: {valid_data_scaled.shape}")
    print(f"test_data_scaled.shape: {test_data_scaled.shape}")
    
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
    
    # Phase 1: Portfolio State Warmup (if needed)
    if train_env.portfolio_normalizer is not None and not train_env.portfolio_normalizer.is_fitted:
        warmup_episodes = train_env.portfolio_normalizer.warmup_episodes
        print(f"\n{'='*60}")
        print(f"PORTFOLIO NORMALIZATION WARMUP PHASE")
        print(f"{'='*60}")
        print(f"Collecting portfolio states for {warmup_episodes} episodes...")
        print("(No training will occur during this phase)")
        
        for warmup_ep in range(warmup_episodes):
            # Simple episode for portfolio state collection only
            state = train_env.reset()
            episode_portfolio_states = []
            
            while True:
                # Random action during warmup (pure exploration)
                action = random.randrange(config.num_actions)
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
        
        print(f"\n✅ Portfolio state collection complete!")
        print(f"🧠 Fitting portfolio normalizer...")
        # The normalizer should now be fitted automatically
        print(f"✅ Ready to start training with normalized portfolio features!")
        print(f"\n{'='*60}")
        print(f"TRAINING PHASE")
        print(f"{'='*60}")
    
    for episode in range(num_episodes):
        # Train one episode
        metrics = agent.train_episode(train_env)
        
        # Store metrics
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        episode_trades.append(metrics['total_trades'])
        episode_invalid_actions.append(metrics['invalid_actions'])
        
        # Print progress
        current_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-1. * agent.steps_done / config.epsilon_decay)
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {metrics['episode_reward']:.4f}")
        print(f"  Return: {metrics['total_return']:.2%}")
        print(f"  Final Value: ${metrics['final_value']:,.2f}")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Winning Trades: {metrics['winning_trades']}")
        print(f"  Losing Trades: {metrics['losing_trades']}")
        print(f"  Invalid Actions: {metrics['invalid_actions']}")
        print(f"  Steps: {metrics['episode_steps']}")
        print(f"  Epsilon: {current_epsilon:.4f} ({'Exploring' if current_epsilon > 0.1 else 'Exploiting'})")
        
        # Portfolio normalizer status (should always be active in training phase)
        if train_env.portfolio_normalizer is not None:
            print(f"  Portfolio Normalizer: ✅ ACTIVE")
        
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
            print(f"  Winning Trades: {val_info['winning_trades']}")
            print(f"  Losing Trades: {val_info['losing_trades']}")
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
            
            # Epsilon-aware early stopping
            current_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-1. * agent.steps_done / config.epsilon_decay)
            
            if patience_counter >= early_stopping_patience and current_epsilon <= EPSILON_EARLY_STOPPING_THRESHOLD:
                print(f"\nEpsilon-aware early stopping triggered at episode {episode+1}")
                print(f"Patience counter: {patience_counter}, Current epsilon: {current_epsilon:.3f}")
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
            elif patience_counter >= early_stopping_patience:
                print(f"\nValidation plateaued but epsilon still high ({current_epsilon:.3f})")
                print(f"Continuing training... (patience reset to {early_stopping_patience // 2})")
                # Partially reset patience counter to give more chances
                patience_counter = early_stopping_patience // 2
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"dqn_checkpoint_episode_{episode}.pt"
            agent.save(checkpoint_path, train_env.portfolio_normalizer)
            print(f"Saved checkpoint at episode {episode}")
            
            # Portfolio normalizer status  
            if train_env.portfolio_normalizer is not None:
                print(f"  Portfolio normalizer: ✅ ACTIVE")
    
    # Multi-day test evaluation
    print("\n" + "="*50)
    print("MULTI-DAY TEST EVALUATION (6 DAYS)")
    print("="*50)
    
    # Run backtests on 6 different days
    num_test_days = min(6, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    all_invalid_action_masks = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        # Reset environment to specific day
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_invalid_action_mask = []
        test_portfolio_values = [config.initial_balance]
        test_price_history = []
        
        while not test_done:
            test_action = agent.select_action(test_state, epsilon=0.0)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            # Track for plotting - store all actions and validity info
            test_action_history.append(test_action)
            test_invalid_action_mask.append(test_info['invalid_action'])
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
        
        # Calculate metrics for this day
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - config.initial_balance) / config.initial_balance
        
        # Calculate performance metrics
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate Sharpe ratio
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            if portfolio_volatility > 1e-8:
                sharpe = test_return / portfolio_volatility
            else:
                sharpe = test_return * 10
        else:
            sharpe = 0.0
        
        # Store results
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
        all_invalid_action_masks.append(test_invalid_action_mask)
        
        # Print day results
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
    sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
    max_drawdowns = [r['max_drawdown'] for r in all_test_results]
    total_trades = [r['total_trades'] for r in all_test_results]
    winning_trades = [r['winning_trades'] for r in all_test_results]
    losing_trades = [r['losing_trades'] for r in all_test_results]
    invalid_actions = [r['invalid_actions'] for r in all_test_results]
    
    print(f"\n{'='*50}")
    print("AGGREGATE TEST RESULTS")
    print(f"{'='*50}")
    print(f"Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
    print(f"Best Return: {np.max(returns):.2%}")
    print(f"Worst Return: {np.min(returns):.2%}")
    print(f"Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    print(f"Average Final Value: ${np.mean(final_values):,.2f}")
    print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
    print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
    print(f"Average Trades per Day: {np.mean(total_trades):.1f}")
    print(f"Average Winning Trades: {np.mean(winning_trades):.1f}")
    print(f"Average Losing Trades: {np.mean(losing_trades):.1f}")
    print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f}")
    
    # Create comprehensive results dictionary
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
            'invalid_action_masks': all_invalid_action_masks,
            'aggregate_stats': {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'win_rate': np.sum([r > 0 for r in returns]) / len(returns),
                'avg_final_value': np.mean(final_values),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_trades': np.mean(total_trades),
                'avg_winning_trades': np.mean(winning_trades),
                'avg_losing_trades': np.mean(losing_trades),
                'avg_invalid_actions': np.mean(invalid_actions)
            }
        },
        'start_date': start_date,
        'end_date': end_date
    }
    
    # Automatically plot the multi-day backtest results
    print(f"\n{'='*50}")
    print("GENERATING MULTI-DAY BACKTEST PLOT")
    print(f"{'='*50}")
    
    try:
        plot_save_path = "multi_day_backtest_results.png"
        plot_multi_day_backtests(results, save_path=plot_save_path, show_plot=False)
        print(f"✅ Multi-day backtest plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"❌ Error generating plot: {e}")
        print("Plot generation failed, but training results are still available")
    
    # Automatically analyze performance and provide recommendations
    # print_trading_analysis(results)
    
    return results


def plot_multi_day_backtests(results: dict, save_path: str = None, show_plot: bool = True):
    """
    Plot portfolio performance for multiple days on the same figure
    
    Args:
        results: Results dictionary from train_dqn function
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Extract multi-day test results
    multi_day_results = results['multi_day_test_results']
    portfolio_values = multi_day_results['portfolio_values']
    individual_days = multi_day_results['individual_days']
    aggregate_stats = multi_day_results['aggregate_stats']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Color palette for different days
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Portfolio Values
    ax1.set_title('Multi-Day Portfolio Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values, individual_days)):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        
        # Create time axis (minutes within trading day)
        time_points = list(range(len(portfolio_vals)))
        
        # Plot portfolio value
        color = colors[i % len(colors)]
        ax1.plot(time_points, portfolio_vals, 
                label=f'Day {day_idx + 1} (Return: {return_pct:.1%})', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final value annotation
        final_val = portfolio_vals[-1]
        ax1.annotate(f'${final_val:,.0f}', 
                    xy=(len(time_points)-1, final_val),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line for initial balance
    initial_balance = results['agent'].config.initial_balance
    ax1.axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5, 
                label=f'Initial Balance (${initial_balance:,.0f})')
    
    ax1.set_xlabel('Minutes into Trading Day')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Normalized Returns (all starting at 100%)
    ax2.set_title('Normalized Returns Comparison (Starting at 100%)', fontsize=14, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values, individual_days)):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        
        # Normalize to percentage returns starting at 100%
        normalized_returns = [(val / portfolio_vals[0]) * 100 for val in portfolio_vals]
        time_points = list(range(len(normalized_returns)))
        
        color = colors[i % len(colors)]
        ax2.plot(time_points, normalized_returns, 
                label=f'Day {day_idx + 1} (Final: {normalized_returns[-1]:.1f}%)', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final percentage annotation
        final_pct = normalized_returns[-1]
        ax2.annotate(f'{final_pct:.1f}%', 
                    xy=(len(time_points)-1, final_pct),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line at 100%
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Break-even (100%)')
    
    ax2.set_xlabel('Minutes into Trading Day')
    ax2.set_ylabel('Portfolio Value (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Add aggregate statistics as text box
    stats_text = f"""Aggregate Statistics (6 Days):
    Average Return: {aggregate_stats['avg_return']:.1%} ± {aggregate_stats['std_return']:.1%}
    Best Return: {aggregate_stats['best_return']:.1%}
    Worst Return: {aggregate_stats['worst_return']:.1%}
    Win Rate: {aggregate_stats['win_rate']:.0%}
    Avg Trades/Day: {aggregate_stats['avg_trades']:.1f}
    Avg Sharpe Ratio: {aggregate_stats['avg_sharpe_ratio']:.2f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def run_standalone_backtest(agent, test_env, num_days: int = 6, plot_results: bool = True, save_plot_path: str = None):
    """
    Run a standalone multi-day backtest with an already trained agent
    
    Args:
        agent: Trained DoubleDuelingDQN agent
        test_env: TradingEnvironment for testing
        num_days: Number of days to test
        plot_results: Whether to plot the results
        save_plot_path: Path to save the plot
    
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*60}")
    print(f"STANDALONE MULTI-DAY BACKTEST ({num_days} DAYS)")
    print(f"{'='*60}")
    
    # Run backtests on multiple days
    num_test_days = min(num_days, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    all_invalid_action_masks = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        # Reset environment to specific day
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_invalid_action_mask = []
        test_portfolio_values = [agent.config.initial_balance]
        test_price_history = []
        
        while not test_done:
            test_action = agent.select_action(test_state, epsilon=0.0)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            # Track for plotting - store all actions and validity info
            test_action_history.append(test_action)
            test_invalid_action_mask.append(test_info['invalid_action'])
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
        
        # Calculate metrics for this day
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - agent.config.initial_balance) / agent.config.initial_balance
        
        # Calculate performance metrics
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate Sharpe ratio
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            if portfolio_volatility > 1e-8:
                sharpe = test_return / portfolio_volatility
            else:
                sharpe = test_return * 10
        else:
            sharpe = 0.0
        
        # Store results
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
        
        # Print day results
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
    sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
    max_drawdowns = [r['max_drawdown'] for r in all_test_results]
    total_trades = [r['total_trades'] for r in all_test_results]
    winning_trades = [r['winning_trades'] for r in all_test_results]
    losing_trades = [r['losing_trades'] for r in all_test_results]
    invalid_actions = [r['invalid_actions'] for r in all_test_results]
    
    print(f"\n{'='*50}")
    print("AGGREGATE BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
    print(f"Best Return: {np.max(returns):.2%}")
    print(f"Worst Return: {np.min(returns):.2%}")
    print(f"Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    print(f"Average Final Value: ${np.mean(final_values):,.2f}")
    print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
    print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
    print(f"Average Trades per Day: {np.mean(total_trades):.1f}")
    print(f"Average Winning Trades: {np.mean(winning_trades):.1f}")
    print(f"Average Losing Trades: {np.mean(losing_trades):.1f}")
    print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f}")
    
    # Create results dictionary
    results = {
        'agent': agent,
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
                'avg_final_value': np.mean(final_values),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_trades': np.mean(total_trades),
                'avg_winning_trades': np.mean(winning_trades),
                'avg_losing_trades': np.mean(losing_trades),
                'avg_invalid_actions': np.mean(invalid_actions)
            }
        }
    }
    
    # Plot results if requested
    if plot_results:
        plot_multi_day_backtests(results, save_path=save_plot_path)
    
    return results