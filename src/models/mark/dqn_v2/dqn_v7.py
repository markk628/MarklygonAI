"""
DQN v7: Enhanced Action Space + Action Masking + Exploration Bonuses
==================================================================

KEY IMPROVEMENTS TO REDUCE INVALID ACTIONS:
1. ENHANCED ACTION SPACE (7 actions instead of 3):
   - 0: HOLD
   - 1: BUY_SMALL (25% of available cash)
   - 2: BUY_MEDIUM (50% of available cash)  
   - 3: BUY_LARGE (75% of available cash)
   - 4: SELL_SMALL (25% of position)
   - 5: SELL_MEDIUM (50% of position)
   - 6: SELL_LARGE (100% of position)

2. ACTION MASKING: Instead of penalizing invalid actions, mask them out
   during action selection so agent can only choose valid actions.

3. EXPLORATION BONUSES: Small rewards for trying new actions to encourage
   more active trading and reduce over-conservative behavior.

4. IMPROVED STATE REPRESENTATION: Better position/cash ratio features
   to help agent understand available opportunities.

5. FIXED WIN/LOSS TRACKING: All sell actions now count towards win/loss 
   statistics, not just full position closes.

6. ANTI-OVERTRADING MEASURES: Trade cooldowns, minimum trade sizes, and 
   moderate transaction cost increases to prevent micro-trading exploitation.

This should convert ~260 invalid actions per day into profitable trades.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
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


class EnhancedTradingConfig(TradingConfig):
    """Enhanced configuration with 7-action space"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override action space
        self.num_actions = 7  # Enhanced action space
        
        # CRITICAL: Update portfolio features count for enhanced state
        self.num_portfolio_features = 30  # Enhanced from 13 to 30 features
        
        # Update total features count
        base_stock_features = len(STOCK_FEATURES_V2)
        self.num_features = base_stock_features + self.num_portfolio_features
        
        # Action sizing ratios
        self.buy_small_ratio = 0.25   # 25% of available cash
        self.buy_medium_ratio = 0.50  # 50% of available cash  
        self.buy_large_ratio = 0.75   # 75% of available cash
        
        self.sell_small_ratio = 0.25  # 25% of position
        self.sell_medium_ratio = 0.50 # 50% of position
        self.sell_large_ratio = 1.00  # 100% of position
        
        # Exploration bonuses (reduced to prevent reward hacking)
        self.exploration_bonus = 0.0006023688699535673  # Much smaller bonus for active trading
        self.position_change_bonus = 0.00026943120061725153  # Smaller bonus for changing position size
        self.max_exploration_per_episode = 0.004641094476076944  # Cap total exploration bonus per episode
        
        # Action masking parameters
        self.use_action_masking = True
        self.min_trade_ratio = 0.020313248479951695  # Minimum 5% of balance for trades (increased from 1%)
        
        # Anti-overtrading measures
        self.min_steps_between_trades = 7  # Minimum 10 steps between trades (increased from 5)
        self.transaction_cost_multiplier = 1.001302240341537  # Moderate increase (reduced from 3.0)
        self.min_position_change_pct = 0.05605805352483136  # Minimum 2% position change for trades
        
        # Enhanced state features
        self.enhanced_state_features = True


class PrioritizedReplayBufferGPU:
    """PER VRAM version - same as v5 but updated for 7 actions"""
    
    def __init__(self, capacity: int, config: EnhancedTradingConfig, device: torch.device=DEVICE):
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


class EnhancedTradingEnvironment:
    """Enhanced trading environment with 7-action space and action masking"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 scaled_data: pd.DataFrame, 
                 config: EnhancedTradingConfig, 
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
        
        print(f"EnhancedTradingEnvironment initialized:")
        print(f"  Minutes per day: {self.minutes_per_day}")
        print(f"  Total trading days: {self.total_days}")
        print(f"  Action space: 7 actions (enhanced)")
        print(f"  Action masking: {'enabled' if config.use_action_masking else 'disabled'}")
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
        
        # Enhanced tracking for 7-action space
        self.action_counts = [0] * 7  # Track usage of each action
        self.successful_trades = [0] * 7  # Track successful trades per action
        self.last_position_change_step = -1
        
        # Portfolio state normalization
        self.portfolio_normalizer = None
        if config.use_portfolio_normalization:
            self.portfolio_normalizer = PortfolioStateNormalizer(
                warmup_episodes=config.portfolio_warmup_episodes,
                update_frequency=config.portfolio_update_frequency
            )
        
        self.reset()
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions for current state - KEY FEATURE FOR ACTION MASKING"""
        valid_actions = [0]  # HOLD is always valid
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Check BUY actions (1, 2, 3) - with stricter requirements
        available_cash = self.balance
        min_investment = self.config.initial_balance * self.config.min_trade_ratio
        
        # Only allow buying if we have sufficient cash and haven't traded recently
        steps_since_last_trade = self.current_step - self.last_trade_step
        can_trade = steps_since_last_trade >= self.config.min_steps_between_trades
        
        if can_trade and available_cash > min_investment:
            
            # BUY_SMALL (25% of available cash) - but check minimum investment
            small_investment = available_cash * self.config.buy_small_ratio
            if small_investment >= min_investment:
                shares_small = small_investment / current_price
                # Apply higher transaction costs to discourage overtrading
                cost_small = shares_small * current_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
                if cost_small <= available_cash:
                    valid_actions.append(1)
            
            # BUY_MEDIUM (50% of available cash)
            medium_investment = available_cash * self.config.buy_medium_ratio
            if medium_investment >= min_investment:
                shares_medium = medium_investment / current_price
                cost_medium = shares_medium * current_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
                if cost_medium <= available_cash:
                    valid_actions.append(2)
            
            # BUY_LARGE (75% of available cash)
            large_investment = available_cash * self.config.buy_large_ratio
            if large_investment >= min_investment:
                shares_large = large_investment / current_price
                cost_large = shares_large * current_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
                if cost_large <= available_cash:
                    valid_actions.append(3)
        
        # Check SELL actions (4, 5, 6) - with stricter requirements
        if self.position > 0 and can_trade:
            position_value = self.position * current_price
            min_sell_value = self.config.initial_balance * self.config.min_trade_ratio
            
            # SELL_SMALL (25% of position) - must meet minimum value and position change
            small_sell_shares = self.position * self.config.sell_small_ratio
            small_sell_value = small_sell_shares * current_price
            position_change_pct = small_sell_shares / self.position
            if (small_sell_value >= min_sell_value and 
                position_change_pct >= self.config.min_position_change_pct):
                valid_actions.append(4)
            
            # SELL_MEDIUM (50% of position)
            medium_sell_shares = self.position * self.config.sell_medium_ratio
            medium_sell_value = medium_sell_shares * current_price
            position_change_pct = medium_sell_shares / self.position
            if (medium_sell_value >= min_sell_value and 
                position_change_pct >= self.config.min_position_change_pct):
                valid_actions.append(5)
            
            # SELL_LARGE (100% of position) - always valid if we have position (full exit)
            valid_actions.append(6)
        
        return valid_actions
    
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
        self.position_entry_step = -1
        self.consecutive_invalid_actions = 0
        self.consecutive_holds = 0
        self.last_action = 0
        self.unrealized_pnl = 0.0
        
        # Enhanced tracking for 7-action space
        self.action_counts = [0] * 7
        self.successful_trades = [0] * 7
        self.last_position_change_step = -1
        self.episode_exploration_bonus = 0.0  # Reset exploration bonus tracking
        
        # Enhanced trading discipline tracking
        self.last_trade_step = -100
        self.last_trade_was_loss = False
        self.steps_since_last_loss = 0
        
        # Initialize portfolio tracking for rewards
        self.last_portfolio_value = self.config.initial_balance
        
        # Select which day to trade
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
        
        # Ensure there is enough data for the window
        min_start = max(self.episode_start, self.config.window_size - 1)
        
        if min_start >= self.episode_end:
            raise ValueError(f"Episode {self.current_day} too short for window size {self.config.window_size}")
            
        self.current_step = min_start
        self.episode_steps_remaining = self.episode_end - self.current_step
        
        if self.episode_steps_remaining < 10:
            print(f"Warning: Very short episode {self.current_day}: only {self.episode_steps_remaining} steps")
        
        # Initialize episode portfolio states
        self.episode_portfolio_states = []
        
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        """Get current state with enhanced features for 7-action space"""
        # Get historical data
        start_idx = max(0, self.current_step - self.config.window_size + 1)
        end_idx = self.current_step + 1
        
        # If not enough historical data, pad with the earliest available data
        if start_idx < self.episode_start - self.config.window_size + 1:
            padding_needed = (self.episode_start - self.config.window_size + 1) - start_idx
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
            if padding_needed > 0:
                first_row = stock_data[0:1]
                padding = np.repeat(first_row, padding_needed, axis=0)
                stock_data = np.vstack([padding, stock_data])
        else:
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
        
        # Ensure exactly window_size rows
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
        
        # Calculate intraday time features
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        time_of_day_normalized = minutes_into_day / self.minutes_per_day
        
        # Market session features
        morning_session = 1.0 if minutes_into_day < 120 else 0.0  # (9:30-11:30)
        midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0  # (11:30-2:00)
        afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0  # (2:00-4:00)
        
        # Position timing features
        position_holding_time = (self.current_step - self.position_entry_step) if self.position_entry_step >= 0 else 0
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0
        
        # ENHANCED STATE FEATURES FOR 7-ACTION SPACE
        # Calculate available action ratios for better decision making
        available_cash = self.balance
        position_value = self.position * current_price if self.position > 0 else 0
        
        # Cash utilization ratios (what % of available cash each buy action would use)
        cash_ratio_small = min(1.0, (available_cash * self.config.buy_small_ratio) / max(available_cash, 1)) if available_cash > 0 else 0
        cash_ratio_medium = min(1.0, (available_cash * self.config.buy_medium_ratio) / max(available_cash, 1)) if available_cash > 0 else 0
        cash_ratio_large = min(1.0, (available_cash * self.config.buy_large_ratio) / max(available_cash, 1)) if available_cash > 0 else 0
        
        # Position utilization ratios (what % of position each sell action would use)
        pos_ratio_small = self.config.sell_small_ratio if self.position > 0 else 0
        pos_ratio_medium = self.config.sell_medium_ratio if self.position > 0 else 0
        pos_ratio_large = self.config.sell_large_ratio if self.position > 0 else 0
        
        # Action opportunity indicators (binary flags)
        valid_actions = self.get_valid_actions()
        can_buy_small = 1.0 if 1 in valid_actions else 0.0
        can_buy_medium = 1.0 if 2 in valid_actions else 0.0
        can_buy_large = 1.0 if 3 in valid_actions else 0.0
        can_sell_small = 1.0 if 4 in valid_actions else 0.0
        can_sell_medium = 1.0 if 5 in valid_actions else 0.0
        can_sell_large = 1.0 if 6 in valid_actions else 0.0
        
        # Trading activity features
        steps_since_last_position_change = (self.current_step - self.last_position_change_step) if self.last_position_change_step >= 0 else 0
        
        # ENHANCED PORTFOLIO FEATURES (30 features total)
        portfolio_features = [
            # Basic portfolio features (0-5)
            self.balance,                    # 0: Raw balance
            position_value,                  # 1: Raw position value
            portfolio_value,                 # 2: Raw portfolio value
            position_ratio,                  # 3: Position ratio (0-1)
            self.unrealized_pnl,            # 4: Raw unrealized P&L
            float(position_holding_time),    # 5: Raw holding time
            
            # Time features (6-9)
            time_of_day_normalized,          # 6: Time of day (0-1)
            morning_session,                 # 7: Morning session binary
            midday_session,                  # 8: Midday session binary
            afternoon_session,               # 9: Afternoon session binary
            
            # Cash utilization features (10-12)
            cash_ratio_small,                # 10: Small buy ratio
            cash_ratio_medium,               # 11: Medium buy ratio
            cash_ratio_large,                # 12: Large buy ratio
            
            # Position utilization features (13-15)
            pos_ratio_small,                 # 13: Small sell ratio
            pos_ratio_medium,                # 14: Medium sell ratio
            pos_ratio_large,                 # 15: Large sell ratio
            
            # Action opportunity flags (16-21)
            can_buy_small,                   # 16: Can execute buy small
            can_buy_medium,                  # 17: Can execute buy medium
            can_buy_large,                   # 18: Can execute buy large
            can_sell_small,                  # 19: Can execute sell small
            can_sell_medium,                 # 20: Can execute sell medium
            can_sell_large,                  # 21: Can execute sell large
            
            # Trading activity features (22-29)
            float(steps_since_last_position_change), # 22: Steps since position change
            float(self.total_trades),        # 23: Total trades this episode
            float(self.winning_trades),      # 24: Winning trades
            float(self.losing_trades),       # 25: Losing trades
            float(len(valid_actions)),       # 26: Number of valid actions available
            float(self.last_action),         # 27: Last action taken
            float(self.consecutive_holds),   # 28: Consecutive holds
            float(self.invalid_actions),     # 29: Invalid actions (should be minimal with masking)
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
        portfolio_state_repeated = portfolio_state.unsqueeze(0).repeat(self.config.window_size, 1)
        
        # Concatenate stock data and portfolio features
        combined_state = torch.cat([stock_data_state, portfolio_state_repeated], dim=1)
        
        return combined_state
    
    def _execute_action(self, action: int, current_price: float) -> Tuple[bool, float, str]:
        """Execute the given action and return (trade_executed, reward_bonus, action_description)"""
        trade_executed = False
        reward_bonus = 0.0
        action_description = ""
        
        # Helper function to calculate capped exploration bonus with cooldown
        def get_exploration_bonus(base_bonus: float) -> float:
            # Check if enough time has passed since last trade to prevent overtrading
            steps_since_last_trade = self.current_step - self.last_trade_step
            if steps_since_last_trade < self.config.min_steps_between_trades:
                return 0.0  # No bonus if trading too frequently
                
            if self.episode_exploration_bonus < self.config.max_exploration_per_episode:
                actual_bonus = min(base_bonus, self.config.max_exploration_per_episode - self.episode_exploration_bonus)
                self.episode_exploration_bonus += actual_bonus
                return actual_bonus
            return 0.0
        
        if action == 0:  # HOLD
            action_description = "HOLD"
            self.consecutive_holds += 1
            
        elif action == 1:  # BUY_SMALL (25% of available cash)
            investment_amount = self.balance * self.config.buy_small_ratio
            shares_to_buy = investment_amount / current_price
            # Apply higher transaction costs to discourage overtrading
            cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            
            if cost <= self.balance:
                self.position += shares_to_buy
                self.balance -= cost
                if self.position_entry_step == -1:  # First position
                    self.entry_price = current_price
                    self.position_entry_step = self.current_step
                else:  # Adding to position
                    # Update weighted average entry price
                    old_value = (self.position - shares_to_buy) * self.entry_price
                    new_value = shares_to_buy * current_price
                    self.entry_price = (old_value + new_value) / self.position
                
                trade_executed = True
                reward_bonus = get_exploration_bonus(self.config.exploration_bonus)  # Capped bonus
                action_description = f"BUY_SMALL ({shares_to_buy:.1f} shares, ${cost:.2f})"
                self.last_position_change_step = self.current_step
                self.last_trade_step = self.current_step  # Update trade cooldown
                self.consecutive_holds = 0
                
        elif action == 2:  # BUY_MEDIUM (50% of available cash)
            investment_amount = self.balance * self.config.buy_medium_ratio
            shares_to_buy = investment_amount / current_price
            cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            
            if cost <= self.balance:
                self.position += shares_to_buy
                self.balance -= cost
                if self.position_entry_step == -1:
                    self.entry_price = current_price
                    self.position_entry_step = self.current_step
                else:
                    old_value = (self.position - shares_to_buy) * self.entry_price
                    new_value = shares_to_buy * current_price
                    self.entry_price = (old_value + new_value) / self.position
                
                trade_executed = True
                # Only give position change bonus for medium/large trades, and cap it
                base_bonus = self.config.exploration_bonus + self.config.position_change_bonus
                reward_bonus = get_exploration_bonus(base_bonus)
                action_description = f"BUY_MEDIUM ({shares_to_buy:.1f} shares, ${cost:.2f})"
                self.last_position_change_step = self.current_step
                self.last_trade_step = self.current_step  # Update trade cooldown
                self.consecutive_holds = 0
                
        elif action == 3:  # BUY_LARGE (75% of available cash)
            investment_amount = self.balance * self.config.buy_large_ratio
            shares_to_buy = investment_amount / current_price
            cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            
            if cost <= self.balance:
                self.position += shares_to_buy
                self.balance -= cost
                if self.position_entry_step == -1:
                    self.entry_price = current_price
                    self.position_entry_step = self.current_step
                else:
                    old_value = (self.position - shares_to_buy) * self.entry_price
                    new_value = shares_to_buy * current_price
                    self.entry_price = (old_value + new_value) / self.position
                
                trade_executed = True
                # Larger bonus for large trades, but still capped
                base_bonus = self.config.exploration_bonus + self.config.position_change_bonus * 1.5
                reward_bonus = get_exploration_bonus(base_bonus)
                action_description = f"BUY_LARGE ({shares_to_buy:.1f} shares, ${cost:.2f})"
                self.last_position_change_step = self.current_step
                self.last_trade_step = self.current_step  # Update trade cooldown
                self.consecutive_holds = 0
                
        elif action == 4:  # SELL_SMALL (25% of position)
            shares_to_sell = self.position * self.config.sell_small_ratio
            revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            
            if shares_to_sell > 0:
                self.balance += revenue
                self.position -= shares_to_sell
                
                # Calculate partial profit (use higher costs for consistent accounting)
                cost_basis = shares_to_sell * self.entry_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
                profit = revenue - cost_basis
                
                if self.position <= 0:  # Closed entire position
                    self.position_entry_step = -1
                
                trade_executed = True
                # Small sell - minimal bonus
                reward_bonus = get_exploration_bonus(self.config.exploration_bonus)
                action_description = f"SELL_SMALL ({shares_to_sell:.1f} shares, ${revenue:.2f}, P&L: ${profit:.2f})"
                self.last_position_change_step = self.current_step
                self.last_trade_step = self.current_step  # Update trade cooldown
                self.consecutive_holds = 0
                
                # Track partial trade P&L AND count as win/loss
                self.total_trades += 1  # Count all sells as completed trades
                if profit > 0:
                    self.winning_trades += 1
                    self.total_profit += profit
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(profit)
                
        elif action == 5:  # SELL_MEDIUM (50% of position)
            shares_to_sell = self.position * self.config.sell_medium_ratio
            revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            
            if shares_to_sell > 0:
                self.balance += revenue
                self.position -= shares_to_sell
                
                cost_basis = shares_to_sell * self.entry_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
                profit = revenue - cost_basis
                
                if self.position <= 0:
                    self.position_entry_step = -1
                
                trade_executed = True
                # Medium sell - moderate bonus, capped
                base_bonus = self.config.exploration_bonus + self.config.position_change_bonus
                reward_bonus = get_exploration_bonus(base_bonus)
                action_description = f"SELL_MEDIUM ({shares_to_sell:.1f} shares, ${revenue:.2f}, P&L: ${profit:.2f})"
                self.last_position_change_step = self.current_step
                self.last_trade_step = self.current_step  # Update trade cooldown
                self.consecutive_holds = 0
                
                # Count all sells as completed trades
                self.total_trades += 1
                if profit > 0:
                    self.winning_trades += 1
                    self.total_profit += profit
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(profit)
                    
        elif action == 6:  # SELL_LARGE (100% of position)
            shares_to_sell = self.position
            revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            
            if shares_to_sell > 0:
                cost_basis = shares_to_sell * self.entry_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
                profit = revenue - cost_basis
                
                self.balance += revenue
                self.position = 0
                self.position_entry_step = -1
                
                self.total_trades += 1
                if profit > 0:
                    self.winning_trades += 1
                    self.total_profit += profit
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(profit)
                
                trade_executed = True
                # Large sell - highest bonus for closing position, but capped
                base_bonus = self.config.exploration_bonus + self.config.position_change_bonus * 2
                reward_bonus = get_exploration_bonus(base_bonus)
                action_description = f"SELL_LARGE ({shares_to_sell:.1f} shares, ${revenue:.2f}, P&L: ${profit:.2f})"
                self.last_position_change_step = self.current_step
                self.last_trade_step = self.current_step  # Update trade cooldown
                self.consecutive_holds = 0
        
        # Update action tracking
        self.action_counts[action] += 1
        if trade_executed:
            self.successful_trades[action] += 1
        
        return trade_executed, reward_bonus, action_description
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state, reward, done, info
        
        ENHANCED REWARD SYSTEM with Action Masking:
        - Portfolio value change (consistent with v5)
        - Exploration bonuses for active trading
        - NO invalid action penalties (actions are masked)
        """
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Store current portfolio value for comparison
        if not hasattr(self, 'last_portfolio_value'):
            self.last_portfolio_value = self.balance + (self.position * current_price)
        
        current_portfolio_value = self.balance + (self.position * current_price)
        
        # PORTFOLIO TRACKING: Reward = portfolio value change (same as v5)
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.11509182887901824)
        reward = portfolio_change * portfolio_scaling
        
        # Execute the action (with action masking, all actions should be valid)
        trade_executed, exploration_bonus, action_description = self._execute_action(action, current_price)
        
        # Add exploration bonus for active trading
        reward += exploration_bonus
        
        # Update last portfolio value for next step
        self.last_portfolio_value = self.balance + (self.position * current_price)
        
        # Move to next step
        self.current_step += 1
        self.episode_steps_remaining -= 1
        
        # Check if episode is done
        done = (self.current_step >= self.episode_end or 
                self.episode_steps_remaining <= 0 or 
                self.balance <= 0)
        
        # Force close any open positions at end of day (with higher transaction costs)
        if done and self.position > 0:
            revenue = self.position * current_price * (1 - self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent * self.config.transaction_cost_multiplier)
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
            
            # Update portfolio value after forced close for final reward
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
        
        # Additional info with enhanced tracking
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'action_description': action_description,
            'invalid_action': False,  # Should always be False with action masking
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
            'exploration_bonus': exploration_bonus,
            'action_counts': self.action_counts.copy(),
            'successful_trades': self.successful_trades.copy(),
            'valid_actions_count': len(self.get_valid_actions()),
            'final_reward': reward
        }
        
        return next_state, reward, done, info

    def render(self):
        # Implement the logic to render the environment
        # This is a placeholder and should be replaced with the actual implementation
        pass

    def close(self):
        # Implement the logic to close the environment
        # This is a placeholder and should be replaced with the actual implementation
        pass 

class EnhancedDoubleDuelingDQN:
    """Enhanced Double Dueling DQN Agent with Action Masking for 7-action space"""
    
    def __init__(self, config: EnhancedTradingConfig, device: torch.device=DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        print(f"Using architecture: {config.architecture_type.value}")
        print(f"Action space: {config.num_actions} actions (enhanced)")
        print(f"Action masking: {'enabled' if config.use_action_masking else 'disabled'}")
        
        # Networks using factory function (updated for 7 actions)
        self.q_network = create_network(config).to(device)
        self.target_network = create_network(config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Print network size for debugging
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        print(f"Network parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                    lr=config.learning_rate, 
                                    weight_decay=1e-5)
        
        # Learning rate scheduler
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
        
    def select_action(self, state: torch.Tensor, valid_actions: List[int] = None, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy with ACTION MASKING"""
        if epsilon is None:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if random.random() > epsilon:
            self.q_network.eval()
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)[0]  # Shape: [num_actions]
                
                # ACTION MASKING: Only consider valid actions
                if valid_actions is not None and len(valid_actions) > 0:
                    # Create a mask for valid actions
                    action_mask = torch.full_like(q_values, float('-inf'))
                    action_mask[valid_actions] = 0  # Set valid actions to 0 (no penalty)
                    
                    # Apply mask to Q-values
                    masked_q_values = q_values + action_mask
                    
                    # Select best valid action
                    action = torch.argmax(masked_q_values).item()
                else:
                    # Fallback: no masking (should not happen with proper environment)
                    action = torch.argmax(q_values).item()
                
            self.q_network.train()
            return action
        else:
            # Random exploration: choose randomly from valid actions
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            else:
                return random.randrange(self.config.num_actions)
    
    def update(self) -> Dict[str, float]:
        """Perform one update step - same as v5 but handles 7 actions"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        try:
            # Calculate current beta for importance sampling
            beta = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * min(1.0, self.steps_done / 100000)
            
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
                torch.cuda.empty_cache()
                return {}
            else:
                raise
    
    def train_episode(self, env: EnhancedTradingEnvironment) -> Dict[str, float]:
        """Train for one episode with enhanced action space"""
        
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_trades = 0
        episode_exploration_bonus = 0
        
        # Track update metrics across the episode
        update_metrics = {
            'loss': [],
            'mean_q': [],
            'mean_td_error': []
        }
        
        # Track action usage
        action_usage = [0] * self.config.num_actions
        
        while True:
            # Get valid actions for current state
            valid_actions = env.get_valid_actions()
            
            # Select action with masking
            action = self.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, done)
            
            # Update counters and tracking
            episode_reward += reward
            episode_steps += 1
            episode_exploration_bonus += info.get('exploration_bonus', 0)
            # Only count meaningful trades (position changes), not all executed actions
            if info.get('trade_executed', False) and info.get('action_description', '') != 'HOLD':
                episode_trades += 1
            
            action_usage[action] += 1
            self.steps_done += 1
            
            # Perform update every update_frequency steps
            if self.steps_done % self.config.update_frequency == 0:
                update_info = self.update()
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
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        
        # Average the update metrics over the episode
        avg_update_metrics = {}
        for key, values in update_metrics.items():
            if values:
                avg_update_metrics[key] = sum(values) / len(values)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'episode_trades': episode_trades,
            'episode_exploration_bonus': episode_exploration_bonus,
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': info['total_trades'],
            'winning_trades': info['winning_trades'],
            'losing_trades': info['losing_trades'],
            'invalid_actions': info['invalid_actions'],  # Should be 0 with masking
            'action_usage': action_usage,
            'action_counts': info.get('action_counts', [0] * 7),
            'successful_trades': info.get('successful_trades', [0] * 7),
            'valid_actions_count': info.get('valid_actions_count', 1),
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
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.update_count = checkpoint['update_count']
        
        # Load portfolio normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.load(normalizer_path)


# Helper functions from v5 (updated for enhanced environment)
def filter_to_regular_hours(df):
    """Filter dataframe to regular market hours using UTC timestamps"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    eastern_times = df['timestamp'].dt.tz_convert('US/Eastern')
    market_open = eastern_times.dt.time >= time(9, 30)
    market_close = eastern_times.dt.time < time(16, 0)
    
    filtered_df = df[market_open & market_close].reset_index(drop=True)
    filtered_df['timestamp'] = filtered_df['timestamp'].dt.tz_localize(None)
    
    print(f"Data filtered: {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df

def load_stock_data(data_path: str, cutoff: pd.Timestamp | None=None, cols_to_keep: list[str]=STOCK_FEATURES_V2) -> tuple[pd.DataFrame, datetime, datetime]:
    """Get saved csv data and filter to regular market hours"""
    df = pd.read_csv(data_path)

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
        
    df = filter_to_regular_hours(df)
    
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
        
    return df[cols_to_keep], start_date, end_date


def train_enhanced_dqn(data_path: str,
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
    Main training function for Enhanced DQN v7 with 7-action space and action masking
    """    
    # Load data
    print("Loading data for Enhanced DQN v7...")
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
            preprocessor_save_path = f"preprocessor_{data_name}_v7.pkl"
        
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
    
    # Initialize enhanced configuration
    config = EnhancedTradingConfig(architecture_type=architecture_type)
    
    # Create enhanced environments
    train_env = EnhancedTradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = EnhancedTradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = EnhancedTradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    print(f"train_data_scaled.shape: {train_data_scaled.shape}")
    print(f"valid_data_scaled.shape: {valid_data_scaled.shape}")
    print(f"test_data_scaled.shape: {test_data_scaled.shape}")
    
    # Create enhanced agent
    agent = EnhancedDoubleDuelingDQN(config)
    
    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    episode_invalid_actions = []
    episode_exploration_bonuses = []
    episode_action_usage = []
    
    validation_rewards = []
    validation_returns = []
    validation_trades = []
    validation_invalid_actions = []
    
    best_validation_return = float('-inf')
    patience_counter = 0
    best_model_state = None
    
    print("Starting Enhanced DQN v7 training...")
    
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
                # Random action during warmup (from valid actions)
                valid_actions = train_env.get_valid_actions()
                action = random.choice(valid_actions) if valid_actions else 0
                next_state, reward, done, info = train_env.step(action)
                
                if hasattr(train_env, 'episode_portfolio_states'):
                    episode_portfolio_states.extend(train_env.episode_portfolio_states)
                
                state = next_state
                if done:
                    break
            
            if episode_portfolio_states:
                train_env.portfolio_normalizer.collect_warmup_data(episode_portfolio_states)
            train_env.portfolio_normalizer.increment_episode()
            
            if (warmup_ep + 1) % 10 == 0 or warmup_ep == warmup_episodes - 1:
                progress = (warmup_ep + 1) / warmup_episodes
                print(f"  Warmup progress: {warmup_ep + 1}/{warmup_episodes} ({progress*100:.1f}%)")
        
        print(f"\n✅ Portfolio state collection complete!")
        print(f"🧠 Fitting portfolio normalizer...")
        print(f"✅ Ready to start training with normalized portfolio features!")
        print(f"\n{'='*60}")
        print(f"ENHANCED TRAINING PHASE (7-ACTION SPACE)")
        print(f"{'='*60}")
    
    for episode in range(num_episodes):
        # Train one episode
        metrics = agent.train_episode(train_env)
        
        # Store metrics
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        episode_trades.append(metrics['episode_trades'])
        episode_invalid_actions.append(metrics['invalid_actions'])
        episode_exploration_bonuses.append(metrics['episode_exploration_bonus'])
        episode_action_usage.append(metrics['action_usage'])
        
        # Print enhanced progress
        current_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-1. * agent.steps_done / config.epsilon_decay)
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {metrics['episode_reward']:.4f} (exploration: +{metrics['episode_exploration_bonus']:.4f})")
        print(f"  Return: {metrics['total_return']:.2%}")
        print(f"  Final Value: ${metrics['final_value']:,.2f}")
        print(f"  Episode Trades: {metrics['episode_trades']} | Total Trades: {metrics['total_trades']}")
        print(f"  Winning Trades: {metrics['winning_trades']} | Losing Trades: {metrics['losing_trades']}")
        print(f"  Invalid Actions: {metrics['invalid_actions']} (should be ~0 with masking)")
        print(f"  Valid Actions Available: {metrics['valid_actions_count']:.1f} avg")
        
        # Action usage breakdown
        action_names = ['HOLD', 'BUY_S', 'BUY_M', 'BUY_L', 'SELL_S', 'SELL_M', 'SELL_L']
        action_usage_str = " | ".join([f"{name}:{count}" for name, count in zip(action_names, metrics['action_usage'])])
        print(f"  Action Usage: {action_usage_str}")
        
        print(f"  Steps: {metrics['episode_steps']} | Epsilon: {current_epsilon:.4f}")
        
        # Portfolio normalizer status
        if train_env.portfolio_normalizer is not None:
            print(f"  Portfolio Normalizer: ✅ ACTIVE")
        
        # Validation with enhanced metrics
        if (episode + 1) % validation_frequency == 0:
            print("\nRunning enhanced validation...")
            
            val_state = val_env.reset()
            val_reward = 0
            val_done = False
            val_trades = 0
            val_action_usage = [0] * 7
            val_valid_actions_total = 0
            val_steps = 0
            
            while not val_done:
                valid_actions = val_env.get_valid_actions()
                val_action = agent.select_action(val_state, valid_actions, epsilon=0.0)
                val_next_state, val_r, val_done, val_info = val_env.step(val_action)
                val_reward += val_r
                val_state = val_next_state
                
                if val_info.get('trade_executed', False):
                    val_trades += 1
                val_action_usage[val_action] += 1
                val_valid_actions_total += len(valid_actions)
                val_steps += 1
            
            val_final_value = val_info['balance'] + (val_info['position'] * val_info['current_price'])
            val_return = (val_final_value - config.initial_balance) / config.initial_balance
            
            validation_rewards.append(val_reward)
            validation_returns.append(val_return)
            validation_trades.append(val_trades)
            validation_invalid_actions.append(val_info['invalid_actions'])
            
            print(f"Validation Results:")
            print(f"  Return: {val_return:.2%}")
            print(f"  Final Value: ${val_final_value:,.2f}")
            print(f"  Episode Trades: {val_trades} | Total Trades: {val_info['total_trades']}")
            print(f"  Winning Trades: {val_info['winning_trades']} | Losing Trades: {val_info['losing_trades']}")
            print(f"  Invalid Actions: {val_info['invalid_actions']}")
            print(f"  Avg Valid Actions: {val_valid_actions_total/val_steps:.1f}")
            
            val_action_usage_str = " | ".join([f"{name}:{count}" for name, count in zip(action_names, val_action_usage)])
            print(f"  Action Usage: {val_action_usage_str}")
            
            # Update learning rate scheduler
            agent.scheduler.step(val_return)
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
            
            # Early stopping check
            if val_return > best_validation_return:
                best_validation_return = val_return
                patience_counter = 0
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
            if patience_counter >= early_stopping_patience and current_epsilon <= EPSILON_EARLY_STOPPING_THRESHOLD:
                print(f"\nEpsilon-aware early stopping triggered at episode {episode+1}")
                print(f"Best validation return: {best_validation_return:.2%}")
                
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
                print(f"Continuing training... (patience reset)")
                patience_counter = early_stopping_patience // 2
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"enhanced_dqn_v7_checkpoint_episode_{episode}.pt"
            agent.save(checkpoint_path, train_env.portfolio_normalizer)
            print(f"Saved enhanced checkpoint at episode {episode}")
    
    # Multi-day test evaluation with enhanced metrics
    print("\n" + "="*60)
    print("ENHANCED MULTI-DAY TEST EVALUATION (20 DAYS)")
    print("="*60)
    
    num_test_days = min(20, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    all_action_descriptions = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning enhanced backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_action_descriptions = []
        test_portfolio_values = [config.initial_balance]
        test_price_history = []
        test_action_usage = [0] * 7
        test_valid_actions_total = 0
        test_steps = 0
        
        while not test_done:
            valid_actions = test_env.get_valid_actions()
            test_action = agent.select_action(test_state, valid_actions, epsilon=0.0)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            test_action_history.append(test_action)
            test_action_descriptions.append(test_info.get('action_description', f'Action {test_action}'))
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
            test_action_usage[test_action] += 1
            test_valid_actions_total += len(valid_actions)
            test_steps += 1
        
        # Calculate enhanced metrics
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - config.initial_balance) / config.initial_balance
        
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            sharpe = test_return / portfolio_volatility if portfolio_volatility > 1e-8 else test_return * 10
        else:
            sharpe = 0.0
        
        # Store enhanced results
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
            'episode_reward': test_reward,
            'action_usage': test_action_usage.copy(),
            'avg_valid_actions': test_valid_actions_total / test_steps if test_steps > 0 else 0,
            'exploration_bonus': test_info.get('exploration_bonus', 0)
        }
        
        all_test_results.append(day_results)
        all_portfolio_values.append(test_portfolio_values)
        all_price_histories.append(test_price_history)
        all_action_histories.append(test_action_history)
        all_action_descriptions.append(test_action_descriptions)
        
        # Print enhanced day results
        print(f"  Day {day_idx + 1} Results:")
        print(f"    Final Value: ${test_final_value:,.2f}")
        print(f"    Return: {test_return:.2%}")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        print(f"    Trades: {test_info['total_trades']} (W:{test_info['winning_trades']}, L:{test_info['losing_trades']})")
        print(f"    Invalid Actions: {test_info['invalid_actions']} (should be ~0)")
        print(f"    Avg Valid Actions: {day_results['avg_valid_actions']:.1f}")
        
        day_action_usage_str = " | ".join([f"{name}:{count}" for name, count in zip(action_names, test_action_usage)])
        print(f"    Action Usage: {day_action_usage_str}")
    
    # Calculate enhanced aggregate statistics
    returns = [r['total_return'] for r in all_test_results]
    final_values = [r['final_value'] for r in all_test_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
    max_drawdowns = [r['max_drawdown'] for r in all_test_results]
    total_trades = [r['total_trades'] for r in all_test_results]
    winning_trades = [r['winning_trades'] for r in all_test_results]
    losing_trades = [r['losing_trades'] for r in all_test_results]
    invalid_actions = [r['invalid_actions'] for r in all_test_results]
    
    # Aggregate action usage
    total_action_usage = [0] * 7
    for result in all_test_results:
        for i, count in enumerate(result['action_usage']):
            total_action_usage[i] += count
    
    print(f"\n{'='*60}")
    print("ENHANCED AGGREGATE TEST RESULTS")
    print(f"{'='*60}")
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
    print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f} (should be ~0)")
    
    # Action usage summary
    total_actions = sum(total_action_usage)
    action_percentages = [count/total_actions*100 if total_actions > 0 else 0 for count in total_action_usage]
    print(f"\nAction Usage Distribution:")
    for name, count, pct in zip(action_names, total_action_usage, action_percentages):
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Create comprehensive results dictionary
    results = {
        'agent': agent,
        'preprocessor': preprocessor,
        'portfolio_normalizer': train_env.portfolio_normalizer,  # CRITICAL: Include portfolio normalizer
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_invalid_actions': episode_invalid_actions,
        'episode_exploration_bonuses': episode_exploration_bonuses,
        'episode_action_usage': episode_action_usage,
        'validation_rewards': validation_rewards,
        'validation_returns': validation_returns,
        'validation_trades': validation_trades,
        'validation_invalid_actions': validation_invalid_actions,
        'multi_day_test_results': {
            'individual_days': all_test_results,
            'portfolio_values': all_portfolio_values,
            'price_histories': all_price_histories,
            'action_histories': all_action_histories,
            'action_descriptions': all_action_descriptions,
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
                'avg_invalid_actions': np.mean(invalid_actions),
                'total_action_usage': total_action_usage,
                'action_percentages': action_percentages
            }
        },
        'start_date': start_date,
        'end_date': end_date,
        'config': config  # Include config for analysis
    }
    
    print(f"\n{'='*60}")
    print("✅ Enhanced DQN v7 Training Complete!")
    print(f"{'='*60}")
    print(f"🔧 7-Action Space: HOLD, BUY_S/M/L, SELL_S/M/L")
    print(f"🎯 Action Masking: Enabled (Invalid actions: {np.mean(invalid_actions):.1f}/day)")
    print(f"🚀 Exploration Bonuses: Active")
    print(f"📈 Enhanced State Features: 30 portfolio features")
    print(f"💰 Average Return: {np.mean(returns):.2%}")
    print(f"🏆 Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    
    return results 


def run_standalone_enhanced_backtest(agent: EnhancedDoubleDuelingDQN, 
                                    test_env: EnhancedTradingEnvironment, 
                                    num_days: int = 20, 
                                    plot_results: bool = True, 
                                    save_plot_path: str = None):
    """
    Run a standalone multi-day backtest with Enhanced DQN v7 agent
    
    Args:
        agent: Trained EnhancedDoubleDuelingDQN agent
        test_env: EnhancedTradingEnvironment for testing
        num_days: Number of days to test
        plot_results: Whether to plot the results
        save_plot_path: Path to save the plot
    
    Returns:
        Dictionary with enhanced backtest results
    """
    print(f"\n{'='*60}")
    print(f"STANDALONE ENHANCED BACKTEST ({num_days} DAYS)")
    print(f"{'='*60}")
    
    num_test_days = min(num_days, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    all_action_descriptions = []
    
    action_names = ['HOLD', 'BUY_S', 'BUY_M', 'BUY_L', 'SELL_S', 'SELL_M', 'SELL_L']
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning enhanced backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_action_descriptions = []
        test_portfolio_values = [agent.config.initial_balance]
        test_price_history = []
        test_action_usage = [0] * 7
        test_valid_actions_total = 0
        test_steps = 0
        
        while not test_done:
            valid_actions = test_env.get_valid_actions()
            test_action = agent.select_action(test_state, valid_actions, epsilon=0.0)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            test_action_history.append(test_action)
            test_action_descriptions.append(test_info.get('action_description', f'Action {test_action}'))
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
            test_action_usage[test_action] += 1
            test_valid_actions_total += len(valid_actions)
            test_steps += 1
        
        # Calculate enhanced metrics
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - agent.config.initial_balance) / agent.config.initial_balance
        
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            sharpe = test_return / portfolio_volatility if portfolio_volatility > 1e-8 else test_return * 10
        else:
            sharpe = 0.0
        
        # Store enhanced results
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
            'episode_reward': test_reward,
            'action_usage': test_action_usage.copy(),
            'avg_valid_actions': test_valid_actions_total / test_steps if test_steps > 0 else 0,
            'exploration_bonus': test_info.get('exploration_bonus', 0)
        }
        
        all_test_results.append(day_results)
        all_portfolio_values.append(test_portfolio_values)
        all_price_histories.append(test_price_history)
        all_action_histories.append(test_action_history)
        all_action_descriptions.append(test_action_descriptions)
        
        # Print enhanced day results
        print(f"  Day {day_idx + 1} Results:")
        print(f"    Final Value: ${test_final_value:,.2f}")
        print(f"    Return: {test_return:.2%}")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        print(f"    Trades: {test_info['total_trades']} (W:{test_info['winning_trades']}, L:{test_info['losing_trades']})")
        print(f"    Invalid Actions: {test_info['invalid_actions']} (should be ~0)")
        print(f"    Avg Valid Actions: {day_results['avg_valid_actions']:.1f}")
        
        day_action_usage_str = " | ".join([f"{name}:{count}" for name, count in zip(action_names, test_action_usage)])
        print(f"    Action Usage: {day_action_usage_str}")
    
    # Calculate enhanced aggregate statistics
    returns = [r['total_return'] for r in all_test_results]
    final_values = [r['final_value'] for r in all_test_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
    max_drawdowns = [r['max_drawdown'] for r in all_test_results]
    total_trades = [r['total_trades'] for r in all_test_results]
    winning_trades = [r['winning_trades'] for r in all_test_results]
    losing_trades = [r['losing_trades'] for r in all_test_results]
    invalid_actions = [r['invalid_actions'] for r in all_test_results]
    
    # Aggregate action usage
    total_action_usage = [0] * 7
    for result in all_test_results:
        for i, count in enumerate(result['action_usage']):
            total_action_usage[i] += count
    
    print(f"\n{'='*60}")
    print("ENHANCED AGGREGATE BACKTEST RESULTS")
    print(f"{'='*60}")
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
    print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f} (should be ~0)")
    
    # Action usage summary
    total_actions = sum(total_action_usage)
    action_percentages = [count/total_actions*100 if total_actions > 0 else 0 for count in total_action_usage]
    print(f"\nAction Usage Distribution:")
    for name, count, pct in zip(action_names, total_action_usage, action_percentages):
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Create results dictionary
    results = {
        'agent': agent,
        'multi_day_test_results': {
            'individual_days': all_test_results,
            'portfolio_values': all_portfolio_values,
            'price_histories': all_price_histories,
            'action_histories': all_action_histories,
            'action_descriptions': all_action_descriptions,
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
                'avg_invalid_actions': np.mean(invalid_actions),
                'total_action_usage': total_action_usage,
                'action_percentages': action_percentages
            }
        }
    }
    
    # Plot results if requested
    if plot_results:
        try:
            # Use the plotting function from v5 if available
            from src.models.mark.dqn_v2.dqn_v5 import plot_multi_day_backtests
            plot_multi_day_backtests(results, save_path=save_plot_path)
            print(f"✅ Enhanced backtest plots generated!")
        except ImportError:
            print(f"⚠️ Plotting function not available - install matplotlib to enable plotting")
    
    return results


# Configuration is now automatically handled by EnhancedTradingConfig


# Quick test function to verify the enhanced environment works
def test_enhanced_environment():
    """Quick test function to verify the enhanced environment and agent work correctly"""
    print("Testing Enhanced DQN v7 Environment...")
    
    # Create dummy data for testing
    np.random.seed(42)
    test_data = pd.DataFrame({
        'close': np.random.uniform(100, 200, 1000),
        'open': np.random.uniform(100, 200, 1000),
        'high': np.random.uniform(150, 250, 1000),
        'low': np.random.uniform(50, 150, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Add any other required features to match STOCK_FEATURES_V2
    for feature in STOCK_FEATURES_V2:
        if feature not in test_data.columns:
            test_data[feature] = np.random.uniform(-1, 1, 1000)
    
    # Create config and environment
    config = EnhancedTradingConfig()
    
    env = EnhancedTradingEnvironment(test_data, test_data, config)
    
    # Test basic functionality
    state = env.reset()
    print(f"✅ Environment reset successful. State shape: {state.shape}")
    
    valid_actions = env.get_valid_actions()
    print(f"✅ Valid actions: {valid_actions}")
    
    action = valid_actions[0] if valid_actions else 0
    next_state, reward, done, info = env.step(action)
    print(f"✅ Step successful. Reward: {reward:.4f}, Done: {done}")
    print(f"✅ Action description: {info.get('action_description', 'N/A')}")
    
    # Create agent
    agent = EnhancedDoubleDuelingDQN(config)
    print(f"✅ Agent created successfully")
    
    # Test action selection with masking
    action = agent.select_action(state, valid_actions, epsilon=0.0)
    print(f"✅ Action selection with masking: {action} (from valid: {valid_actions})")
    
    print("\n🎉 Enhanced DQN v7 test completed successfully!")
    return True


if __name__ == "__main__":
    # Run a quick test if this file is executed directly
    test_enhanced_environment() 