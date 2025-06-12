"""
SAC Trading Environments
=======================

Contains all trading environment variants for SAC agents:
1. BasicTradingEnvironment - Simple buy-sell cycles (SAC v1)
2. WeightedAverageTradingEnvironment - Weighted average cost basis (SAC v2)  
3. LotBasedTradingEnvironment - Individual lot tracking (SAC v3)

All environments share the same interface and can be used interchangeably
with any SAC agent configuration.
"""

import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from datetime import datetime, time
from dataclasses import dataclass

from src.config.config import DEVICE, MINUTES_PER_TRADING_DAY
from src.models.mark.dqn_v2.normalization import PortfolioStateNormalizer
from src.models.jeawan.sac.sac_config import SACConfig, TradingMode


# ==========================================
# SHARED TRADING LOT CLASS (for Lot-Based)
# ==========================================

@dataclass
class TradingLot:
    """Represents a single trading lot (buy transaction)"""
    shares: float
    entry_price: float
    timestamp: int
    cost_basis: float = 0.0
    
    def __post_init__(self):
        """Calculate cost basis if not provided"""
        if self.cost_basis == 0:
            # Use config values from environment context
            self.cost_basis = self.shares * self.entry_price * (1 + 0.001)  # Default fee
    
    def sell_shares(self, shares_to_sell: float, sell_price: float, fee_percent: float) -> Tuple[float, float, bool]:
        """Sell shares from this lot and return (profit, remaining_shares, lot_completed)"""
        if shares_to_sell > self.shares:
            shares_to_sell = self.shares
            
        # Calculate profit for sold portion
        revenue = shares_to_sell * sell_price * (1 - fee_percent)
        cost_portion = (shares_to_sell / self.shares) * self.cost_basis
        profit = revenue - cost_portion
        
        # Update lot
        self.shares -= shares_to_sell
        self.cost_basis -= cost_portion
        
        lot_completed = (self.shares < 1e-6)
        if lot_completed:
            self.shares = 0.0
            self.cost_basis = 0.0
            
        return profit, self.shares, lot_completed


# ==========================================
# BASE TRADING ENVIRONMENT
# ==========================================

class BaseTradingEnvironment:
    """Base class with shared functionality for all trading environments"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 scaled_data: pd.DataFrame, 
                 config: SACConfig, 
                 mode: TradingMode = TradingMode.TRAIN, 
                 device: torch.device = DEVICE):
        self.data = data
        self.scaled_data = scaled_data
        self.config = config
        self.mode = mode
        self.device = device
        self.minutes_per_day = config.minutes_per_day
        
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
    
    def _get_base_state(self) -> Tuple[np.ndarray, Dict]:
        """Get base state components shared by all environments"""
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
        
        # Time features
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        time_of_day_normalized = minutes_into_day / self.minutes_per_day
        
        # Session features
        morning_session = 1.0 if minutes_into_day < 120 else 0.0
        midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0
        afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0
        
        return stock_data, {
            'current_price': current_price,
            'time_of_day_normalized': time_of_day_normalized,
            'morning_session': morning_session,
            'midday_session': midday_session,
            'afternoon_session': afternoon_session
        }
    
    def _calculate_reward(self, current_portfolio_value: float) -> float:
        """Calculate reward based on portfolio value change"""
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        portfolio_scaling = getattr(self.config, 'portfolio_scaling', 1.0)
        reward = portfolio_change * portfolio_scaling
        self.last_portfolio_value = current_portfolio_value
        return reward


# ==========================================
# BASIC TRADING ENVIRONMENT (SAC v1)
# ==========================================

class BasicTradingEnvironment(BaseTradingEnvironment):
    """Basic trading environment with simple buy-sell cycles"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"BasicTradingEnvironment initialized:")
        print(f"  Minutes per day: {self.minutes_per_day}")
        print(f"  Total trading days: {self.total_days}")
        print(f"  Trading logic: Simple buy-sell cycles")
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
        """Get current state for basic trading"""
        stock_data, base_info = self._get_base_state()
        current_price = base_info['current_price']
        
        # Calculate portfolio metrics
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
        
        # Position timing
        position_holding_time = (self.current_step - self.position_entry_step) if self.position_entry_step >= 0 else 0
        
        # Position ratio
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0
        
        # Action validity
        can_buy = 1.0 if self.balance > 0 else 0.0
        can_sell = 1.0 if self.position > 0 else 0.0
        
        # Portfolio features
        if self.config.use_portfolio_normalization:
            portfolio_features = [
                self.balance,                     # 0: Raw balance
                self.position * current_price,    # 1: Raw position value  
                portfolio_value,                  # 2: Raw portfolio value
                position_ratio,                   # 3: Position ratio (already 0-1)
                self.unrealized_pnl,              # 4: Raw unrealized P&L
                position_holding_time,            # 5: Raw holding time in steps
                base_info['time_of_day_normalized'], # 6: Time of day (already 0-1)
                base_info['morning_session'],     # 7: Session indicator (0 or 1)
                base_info['midday_session'],      # 8: Session indicator (0 or 1)
                base_info['afternoon_session'],   # 9: Session indicator (0 or 1)
                can_buy,                          # 10: Can buy flag (0 or 1)
                can_sell,                         # 11: Can sell flag (0 or 1)
                self.invalid_actions              # 12: Raw invalid action count
            ]
        else:
            # Manual normalization fallback
            initial_balance = self.config.initial_balance
            normalized_balance = self.balance / initial_balance if initial_balance > 0 else 0
            normalized_position = self.position * current_price / initial_balance if initial_balance > 0 else 0
            normalized_portfolio_value = portfolio_value / initial_balance if initial_balance > 0 else 0
            normalized_holding_time = min(position_holding_time / 60, 1.0)
            recent_invalid_rate = self.invalid_actions / max(self.current_step - self.episode_start + 1, 1)
            
            portfolio_features = [
                normalized_balance,               # 0: Manually normalized balance
                normalized_position,              # 1: Manually normalized position
                normalized_portfolio_value,       # 2: Manually normalized portfolio value
                position_ratio,                   # 3: Position ratio (already 0-1)
                self.unrealized_pnl,              # 4: Raw unrealized P&L
                normalized_holding_time,          # 5: Manually normalized holding time
                base_info['time_of_day_normalized'], # 6: Time of day (already 0-1)
                base_info['morning_session'],     # 7: Session indicator (0 or 1)
                base_info['midday_session'],      # 8: Session indicator (0 or 1) 
                base_info['afternoon_session'],   # 9: Session indicator (0 or 1)
                can_buy,                          # 10: Can buy flag (0 or 1)
                can_sell,                         # 11: Can sell flag (0 or 1)
                min(recent_invalid_rate, 1.0)    # 12: Manually normalized invalid rate
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
    
    def _execute_action(self, action_value: float) -> Tuple[bool, bool]:
        """Execute basic trading action (simple buy-sell cycles)"""
        current_price = self.data.iloc[self.current_step]['close']
        trade_executed = False
        invalid_action = False
        
        # Debug print for testing mode
        if self.mode == TradingMode.TEST and self.current_step % 100 == 0:
            print(f"    DEBUG: Action {action_value:.6f}, Balance ${self.balance:.2f}, Position {self.position:.2f}")
        
        # Only skip truly zero actions
        if abs(action_value) < 1e-6:
            return False, False
        
        if action_value > 0:  # Buy action
            # Can only buy if no existing position (like DQN v5)
            if self.position > 0 or self.balance <= 0:
                invalid_action = True
            else:
                # Calculate buy amount using action magnitude
                cash_to_use = self.balance * action_value * self.config.max_position_size
                shares_to_buy = cash_to_use / current_price
                total_cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                if total_cost <= self.balance:
                    # Execute buy (open new position)
                    self.position = shares_to_buy
                    self.entry_price = current_price
                    self.position_entry_step = self.current_step
                    self.balance -= total_cost
                    trade_executed = True
                else:
                    invalid_action = True
                    
        else:  # Sell action (action_value < 0)
            # Can only sell if holding position (like DQN v5)
            if self.position <= 0:
                invalid_action = True
            else:
                # Calculate sell amount using action magnitude
                sell_proportion = abs(action_value)
                shares_to_sell = self.position * sell_proportion
                revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_percent)
                
                # Execute sell
                self.balance += revenue
                
                # Calculate profit for the sold portion
                cost_basis = shares_to_sell * self.entry_price * (1 + self.config.transaction_fee_percent)
                profit = revenue - cost_basis
                
                # Update position
                self.position -= shares_to_sell
                
                # Track profit/loss
                if profit > 0:
                    self.total_profit += profit
                else:
                    self.total_loss += abs(profit)
                
                # If position is completely closed, count as one completed trade
                if self.position < 1e-6:  # Essentially zero - position fully closed
                    self.position = 0.0
                    self.position_entry_step = -1
                    
                    # Count as one completed trade (buy-sell cycle)
                    self.total_trades += 1
                    
                    # Determine if overall position was profitable
                    if profit > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                
                trade_executed = True
        
        return trade_executed, invalid_action
    
    def step(self, action: float) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Store current portfolio value for reward calculation
        if not hasattr(self, 'last_portfolio_value'):
            self.last_portfolio_value = self.balance + (self.position * current_price)
        
        current_portfolio_value = self.balance + (self.position * current_price)
        
        # Execute action
        trade_executed, invalid_action = self._execute_action(action)
        
        # Track invalid actions
        if invalid_action:
            self.invalid_actions += 1
        
        # Calculate reward based on action validity
        if invalid_action:
            # Invalid actions: Only penalty, no portfolio reward
            # Rationale: Action didn't execute, so shouldn't get credit for market movements
            invalid_penalty = getattr(self.config, 'invalid_penalty', 0.1)
            reward = -invalid_penalty
            
            # Update last_portfolio_value to current for next step, but don't reward the change
            # This prevents "reward accumulation" from market movements during invalid actions
            new_portfolio_value = self.balance + (self.position * current_price)
            self.last_portfolio_value = new_portfolio_value
        else:
            # Valid actions: Calculate portfolio change reward as normal
            new_portfolio_value = self.balance + (self.position * current_price)
            reward = self._calculate_reward(new_portfolio_value)
        
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
            
            # Add to episode totals
            if profit > 0:
                self.total_profit += profit
            else:
                self.total_loss += abs(profit)
            
            # Count the final position close as a completed trade
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Final reward calculation (always applied regardless of last action validity)
            final_portfolio_value = self.balance
            final_change = final_portfolio_value - self.last_portfolio_value
            portfolio_scaling = getattr(self.config, 'portfolio_scaling', 1.0)
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


# ==========================================
# WEIGHTED AVERAGE TRADING ENVIRONMENT (SAC v2)
# ==========================================

class WeightedAverageTradingEnvironment(BasicTradingEnvironment):
    """Trading environment with weighted average cost basis tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"WeightedAverageTradingEnvironment initialized:")
        print(f"  Trading logic: Weighted average cost basis")
        print(f"  Can buy multiple times to build position")
        print(f"  Expected trades per episode: 5-15 position cycles")
        
    def reset(self, day_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment with weighted average cost basis tracking"""
        result = super().reset(day_idx)
        self.weighted_avg_entry_price = 0.0  # Weighted average cost basis
        return result
    
    def _execute_action(self, action_value: float) -> Tuple[bool, bool]:
        """Execute action with weighted average cost basis tracking"""
        current_price = self.data.iloc[self.current_step]['close']
        trade_executed = False
        invalid_action = False
        
        # Debug print for testing mode
        if self.mode == TradingMode.TEST and self.current_step % 100 == 0:
            print(f"    DEBUG: Action {action_value:.6f}, Position {self.position:.2f}, Avg Entry ${self.weighted_avg_entry_price:.2f}")
        
        # Only skip truly zero actions
        if abs(action_value) < 1e-6:
            return False, False
        
        if action_value > 0:  # Buy action
            if self.balance <= 0:
                invalid_action = True
            else:
                # Calculate buy amount
                cash_to_use = self.balance * action_value * self.config.max_position_size
                shares_to_buy = cash_to_use / current_price
                total_cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                if total_cost <= self.balance:
                    # Update weighted average entry price
                    if self.position > 0:
                        # Adding to existing position - calculate weighted average
                        old_value = self.position * self.weighted_avg_entry_price
                        new_value = shares_to_buy * current_price
                        total_shares = self.position + shares_to_buy
                        self.weighted_avg_entry_price = (old_value + new_value) / total_shares
                        self.position = total_shares
                    else:
                        # New position
                        self.position = shares_to_buy
                        self.weighted_avg_entry_price = current_price
                        self.position_entry_step = self.current_step
                    
                    self.balance -= total_cost
                    trade_executed = True
                else:
                    invalid_action = True
                    
        else:  # Sell action
            if self.position <= 0:
                invalid_action = True
            else:
                # Calculate sell amount
                sell_proportion = abs(action_value)
                shares_to_sell = self.position * sell_proportion
                revenue = shares_to_sell * current_price * (1 - self.config.transaction_fee_percent)
                
                # Execute sell
                self.balance += revenue
                
                # Calculate profit using weighted average cost
                cost_basis = shares_to_sell * self.weighted_avg_entry_price * (1 + self.config.transaction_fee_percent)
                profit = revenue - cost_basis
                
                # Update position
                self.position -= shares_to_sell
                
                # Track profit/loss
                if profit > 0:
                    self.total_profit += profit
                else:
                    self.total_loss += abs(profit)
                
                # If position is completely closed, count as one completed trade
                if self.position < 1e-6:
                    self.position = 0.0
                    self.weighted_avg_entry_price = 0.0  # Reset for next position
                    self.position_entry_step = -1
                    
                    # Count as one completed trade cycle
                    self.total_trades += 1
                    if profit > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                
                trade_executed = True
        
        return trade_executed, invalid_action


# ==========================================
# LOT-BASED TRADING ENVIRONMENT (SAC v3)
# ==========================================

class LotBasedTradingEnvironment(BasicTradingEnvironment):
    """Trading environment with individual lot tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"LotBasedTradingEnvironment initialized:")
        print(f"  Trading logic: Individual lot tracking")
        print(f"  Max lots: {self.config.max_lots}")
        print(f"  Lot method: {self.config.lot_method}")
        print(f"  Expected trades per episode: 10-50+ lot completions")
        
    def reset(self, day_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment with lot-based tracking"""
        result = super().reset(day_idx)
        self.lots: List[TradingLot] = []  # List of trading lots
        return result
    
    @property
    def total_position(self) -> float:
        """Calculate total position across all lots"""
        return sum(lot.shares for lot in self.lots)
    
    @property
    def weighted_avg_cost_basis(self) -> float:
        """Calculate weighted average cost basis across all lots"""
        if not self.lots:
            return 0.0
        total_cost = sum(lot.cost_basis for lot in self.lots)
        total_shares = sum(lot.shares for lot in self.lots)
        return total_cost / total_shares if total_shares > 0 else 0.0
    
    def _merge_smallest_lots(self):
        """Merge the two smallest lots to free up space"""
        if len(self.lots) < 2:
            return
            
        # Sort by shares to find smallest lots
        self.lots.sort(key=lambda lot: lot.shares)
        smallest = self.lots[0]
        second_smallest = self.lots[1]
        
        # Merge into the first lot
        total_shares = smallest.shares + second_smallest.shares
        total_cost = smallest.cost_basis + second_smallest.cost_basis
        weighted_price = total_cost / (total_shares * (1 + self.config.transaction_fee_percent))
        
        # Update first lot
        smallest.shares = total_shares
        smallest.entry_price = weighted_price
        smallest.cost_basis = total_cost
        smallest.timestamp = min(smallest.timestamp, second_smallest.timestamp)
        
        # Remove second lot
        self.lots.remove(second_smallest)
    
    def _execute_sell_order(self, shares_to_sell: float, sell_price: float):
        """Execute sell order against lots using FIFO/LIFO method"""
        remaining_to_sell = shares_to_sell
        completed_lots = []
        
        # Sort lots based on method
        if self.config.lot_method == "FIFO":
            lots_to_process = sorted(self.lots, key=lambda lot: lot.timestamp)
        else:  # LIFO
            lots_to_process = sorted(self.lots, key=lambda lot: lot.timestamp, reverse=True)
        
        for lot in lots_to_process:
            if remaining_to_sell <= 0:
                break
                
            if lot.shares <= 0:
                continue
                
            # Sell from this lot
            shares_from_lot = min(remaining_to_sell, lot.shares)
            profit, remaining_shares, lot_completed = lot.sell_shares(
                shares_from_lot, sell_price, self.config.transaction_fee_percent
            )
            
            # Update balance
            revenue = shares_from_lot * sell_price * (1 - self.config.transaction_fee_percent)
            self.balance += revenue
            
            # Track profit/loss
            if profit > 0:
                self.total_profit += profit
            else:
                self.total_loss += abs(profit)
            
            # If lot completed, count as one trade
            if lot_completed:
                completed_lots.append(lot)
                self.total_trades += 1
                
                # Determine if this specific lot was profitable
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            remaining_to_sell -= shares_from_lot
        
        # Remove completed lots
        for completed_lot in completed_lots:
            if completed_lot in self.lots:
                self.lots.remove(completed_lot)
        
        # Update position entry step if no lots remain
        if not self.lots:
            self.position_entry_step = -1
    
    def _execute_action(self, action_value: float) -> Tuple[bool, bool]:
        """Execute action with lot-based tracking"""
        current_price = self.data.iloc[self.current_step]['close']
        trade_executed = False
        invalid_action = False
        
        # Debug print for testing mode
        if self.mode == TradingMode.TEST and self.current_step % 100 == 0:
            print(f"    DEBUG: Action {action_value:.6f}, Lots {len(self.lots)}, Total Position {self.total_position:.2f}")
        
        # Only skip truly zero actions
        if abs(action_value) < 1e-6:
            return False, False
        
        if action_value > 0:  # Buy action
            if self.balance <= 0:
                invalid_action = True
            else:
                # Calculate buy amount
                cash_to_use = self.balance * action_value * self.config.max_position_size
                shares_to_buy = cash_to_use / current_price
                total_cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                if total_cost <= self.balance:
                    # Check lot limit
                    if len(self.lots) >= self.config.max_lots:
                        self._merge_smallest_lots()
                    
                    # Create new trading lot
                    new_lot = TradingLot(
                        shares=shares_to_buy,
                        entry_price=current_price,
                        timestamp=self.current_step,
                        cost_basis=total_cost
                    )
                    self.lots.append(new_lot)
                    
                    self.balance -= total_cost
                    trade_executed = True
                    
                    # Update position entry step for the first lot
                    if self.position_entry_step == -1:
                        self.position_entry_step = self.current_step
                        
                else:
                    invalid_action = True
                    
        else:  # Sell action
            if self.total_position <= 0:
                invalid_action = True
            else:
                # Calculate sell amount
                sell_proportion = abs(action_value)
                shares_to_sell = self.total_position * sell_proportion
                
                # Execute sell using FIFO/LIFO method
                self._execute_sell_order(shares_to_sell, current_price)
                trade_executed = True
        
        return trade_executed, invalid_action
    
    def _get_state(self) -> torch.Tensor:
        """Get current state with lot-based features"""
        stock_data, base_info = self._get_base_state()
        current_price = base_info['current_price']
        
        # Calculate portfolio metrics using lots
        total_position_value = self.total_position * current_price
        portfolio_value = self.balance + total_position_value
        
        # Calculate unrealized P&L across all lots
        if self.lots:
            total_cost_basis = sum(lot.cost_basis for lot in self.lots)
            total_market_value = self.total_position * current_price
            self.unrealized_pnl = (total_market_value - total_cost_basis) / total_cost_basis if total_cost_basis > 0 else 0.0
        else:
            self.unrealized_pnl = 0.0
        
        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        # Position timing (based on oldest lot)
        if self.lots:
            oldest_lot_time = min(lot.timestamp for lot in self.lots)
            position_holding_time = self.current_step - oldest_lot_time
        else:
            position_holding_time = 0
        
        # Position ratio
        position_ratio = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Action validity
        can_buy = 1.0 if self.balance > 0 else 0.0
        can_sell = 1.0 if self.total_position > 0 else 0.0
        
        # Portfolio features including lot-based metrics
        if self.config.use_portfolio_normalization:
            portfolio_features = [
                self.balance,                     # 0: Raw balance
                total_position_value,             # 1: Raw position value (all lots)
                portfolio_value,                  # 2: Raw portfolio value
                position_ratio,                   # 3: Position ratio (already 0-1)
                self.unrealized_pnl,              # 4: Raw unrealized P&L (across all lots)
                position_holding_time,            # 5: Raw holding time (oldest lot)
                base_info['time_of_day_normalized'], # 6: Time of day (already 0-1)
                base_info['morning_session'],     # 7: Session indicator (0 or 1)
                base_info['midday_session'],      # 8: Session indicator (0 or 1)
                base_info['afternoon_session'],   # 9: Session indicator (0 or 1)
                can_buy,                          # 10: Can buy flag (0 or 1)
                can_sell,                         # 11: Can sell flag (0 or 1)
                len(self.lots)                    # 12: Number of active lots
            ]
        else:
            # Manual normalization fallback
            initial_balance = self.config.initial_balance
            normalized_balance = self.balance / initial_balance if initial_balance > 0 else 0
            normalized_position = total_position_value / initial_balance if initial_balance > 0 else 0
            normalized_portfolio_value = portfolio_value / initial_balance if initial_balance > 0 else 0
            normalized_holding_time = min(position_holding_time / 60, 1.0)
            
            portfolio_features = [
                normalized_balance,               # 0: Manually normalized balance
                normalized_position,              # 1: Manually normalized position
                normalized_portfolio_value,       # 2: Manually normalized portfolio value
                position_ratio,                   # 3: Position ratio (already 0-1)
                self.unrealized_pnl,              # 4: Raw unrealized P&L
                normalized_holding_time,          # 5: Manually normalized holding time
                base_info['time_of_day_normalized'], # 6: Time of day (already 0-1)
                base_info['morning_session'],     # 7: Session indicator (0 or 1)
                base_info['midday_session'],      # 8: Session indicator (0 or 1) 
                base_info['afternoon_session'],   # 9: Session indicator (0 or 1)
                can_buy,                          # 10: Can buy flag (0 or 1)
                can_sell,                         # 11: Can sell flag (0 or 1)
                len(self.lots) / self.config.max_lots  # 12: Normalized number of lots
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


# ==========================================
# ENVIRONMENT FACTORY
# ==========================================

def create_environment(data: pd.DataFrame, 
                      scaled_data: pd.DataFrame, 
                      config: SACConfig, 
                      mode: TradingMode = TradingMode.TRAIN, 
                      device: torch.device = DEVICE):
    """Factory function to create trading environments based on config"""
    
    if config.environment_type.value == 'weighted_avg':
        return WeightedAverageTradingEnvironment(data, scaled_data, config, mode, device)
    elif config.environment_type.value == 'lot_based':
        return LotBasedTradingEnvironment(data, scaled_data, config, mode, device)
    else:  # basic
        return BasicTradingEnvironment(data, scaled_data, config, mode, device)


def compare_environments():
    """Compare the three environment types"""
    print("="*60)
    print("TRADING ENVIRONMENT COMPARISON")
    print("="*60)
    print("1. BASIC TRADING ENVIRONMENT (SAC v1)")
    print("   • Simple buy-sell cycles")
    print("   • Can only buy when position = 0")
    print("   • Can only sell when position > 0") 
    print("   • Expected trades: 5-15 per episode")
    print("   • Memory efficient, simple logic")
    print()
    print("2. WEIGHTED AVERAGE TRADING ENVIRONMENT (SAC v2)")
    print("   • Can buy multiple times to build position")
    print("   • Weighted average cost basis tracking")
    print("   • Can sell partial or full positions")
    print("   • Expected trades: 5-15 per episode")
    print("   • Good balance of complexity and realism")
    print()
    print("3. LOT-BASED TRADING ENVIRONMENT (SAC v3)")
    print("   • Each buy creates separate lot")
    print("   • FIFO/LIFO sell targeting")
    print("   • Precise profit attribution per lot")
    print("   • Expected trades: 10-50+ per episode")
    print("   • Most realistic, highest complexity")
    print()
    print("RECOMMENDATION:")
    print("• Start with BASIC for initial testing")
    print("• Use WEIGHTED_AVERAGE for production")
    print("• Use LOT_BASED for advanced strategies")


if __name__ == "__main__":
    compare_environments() 