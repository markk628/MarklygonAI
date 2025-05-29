import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch.types import Number
from typing import Optional

from src.config.config import DATA_DIR, MODELS_DIR, TRAIN_RATIO, VALID_RATIO
from src.utils.utils import format_duration

pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class StockTradingEnv:
    """
    Environment for stock trading
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        initial_balance: int=10000, 
        transaction_fee: float=0.0015, 
        window_size: int=20,
        mode: str='train'  # 'train', 'validation', or 'test'
    ):  
        # TODO might not need reset_index
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self.data_nparray: np.ndarray = data.values
        self.close_prices_idx = data.columns.get_loc('close')
        self.initial_balance: int = initial_balance
        self.transaction_fee: float = transaction_fee
        self.window_size: int = window_size
        self.mode: str = mode
        self.steps_per_episode: int = len(data) - window_size
        self._feature_cache = {}
        
        # trading conditions
        self.min_trade_interval: int = 2     # minimum interval between trades
        self.max_position_size: float = 0.7  # maximum position size (how much capitol can be used per trade)
        self.min_trade_amount: int = 300     #  minimum amount required to make a trade
        
        # reward settings
        self.hold_penalty_base: float = -0.02     # holding penalty
        self.hold_penalty_max: float = -0.15      # maximum holding penalty
        self.profit_reward_weight: int = 50       # profit reward weight (used to scale rewards for profitable trades)
        self.loss_penalty_weight: int = 25        # loss penalty weight (used to scale penalty for loss)
        self.trade_reward: float = 0.2            # trade execution reward
        self.first_trade_reward: float = 0.1      # first trade reward
        self.opposite_action_reward: float = 0.1  # reward for switching actions
        self.invalid_action_penalty: float = 0.5  # penalty for taking invalid action
        self.stop_loss_penalty: float = 0.5

        # profit/loss threshold settings
        self.profit_threshold: float = 0.01      # profit-taking threshold (profits are taken when gains exceed threshold%)
        self.loss_threshold: float = -0.003      # loss-cutting threshold (losses are cut once the position loses threshold% or more)
        self.trailing_stop: float = 0.008        # trailing stop value (trailing stop is activated after an threshold% move)
        self.max_profit_threshold: float = 0.02  # maximum profit-taking threshold
        self.max_loss_threshold: float = -0.008  # maximum loss threshold

        # additional profitability-related settings
        self.profit_taking_levels = [0.01, 0.02, 0.03]  # tiered profit-taking levels
        self.profit_taking_weights = [1.0, 1.5, 2.0]    # reward weights for each profit-taking level
        self.volatility_threshold: float = 0.02         # volatility threshold (used to determine whether to adjust behavior during periods of high volatility)
        self.trend_following_weight: float = 1.2        # trend-following reward weight (encourages the agent to follow the trend by increasing rewards when aligned with it)
        
        # state variables
        self.current_step: int = 0
        self.balance: int = initial_balance
        self.shares_held: int = 0
        self.total_trades: int = 0
        self.total_shares_bought: int = 0
        self.total_shares_sold: int = 0
        self.total_cost: int = 0
        self.consecutive_trades: int = 0
        self.consecutive_holds: int = 0
        self.last_portfolio_value: int = initial_balance
        self.last_action = None
        self.last_trade_step: int = -1
        self.trade_history = []
        self.entry_price: int = 0
        self.max_profit: int = 0
        self.max_loss: int = 0
        self.trailing_stop_price: int = 0
        self.position_open: bool = False
        
        # portfolio metrics tracking
        self.portfolio_values = [self.initial_balance]
        self.action_history = []
        self.price_history = []
        
        self.reset()
        
    def reset(self) -> NDArray:
        # start at window_size to ensure enough historical data for the first state's features
        self.current_step = self.window_size 
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_cost = 0
        self.consecutive_trades = 0
        self.consecutive_holds = 0
        self.last_portfolio_value = self.initial_balance
        self.last_action = None
        self.last_trade_step = -1
        self.trade_history = []
        self.entry_price = 0
        self.max_profit = 0
        self.max_loss = 0
        self.trailing_stop_price = 0
        self.position_open = False
        self.portfolio_values = [self.initial_balance]
        self.action_history = []
        self.price_history = []
        
        return self._get_state()
    
    def _get_features(self, current_idx: int) -> NDArray:
        """
        Extract features from a rolling window of historical data
        """
        # return cached data for current_idx if it exists
        if current_idx in self._feature_cache:
            return self._feature_cache[current_idx]
                
        start_idx: int = current_idx - self.window_size
        end_idx: int = current_idx
        features = self.data_nparray[start_idx:end_idx]
        self._feature_cache[current_idx] = features
        return features

    def _get_state(self) -> NDArray:
        """
        Construct the current state with normalized features and portfolio information
        """
        # features are from the window [self.current_step - self.window_size, self.current_step - 1]
        raw_features = self._get_features(self.current_step)

        # rolling window scaling: Fit and transform on the features of the current window
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(raw_features)
        normalized_features_flattened = normalized_features.flatten()
    
        # current price for portfolio valuation
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        portfolio_value = self.balance + self.shares_held * current_price
        
        normalized_portfolio_value = portfolio_value / self.initial_balance
        
        # Enhanced portfolio information
        balance_ratio = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        shares_value_ratio = (self.shares_held * current_price) / self.initial_balance if self.initial_balance > 0 else 0
        
        # Calculate profit/loss from current position
        avg_buy_price = self.total_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0
        position_pl = (current_price - avg_buy_price) * self.shares_held if self.shares_held > 0 else 0
        position_pl_ratio = position_pl / self.initial_balance if self.initial_balance > 0 else 0

        is_holding_shares = float(self.shares_held > 0)

        normalized_shares_held = self.shares_held / self.max_position_size
        normalized_total_trades = self.total_trades / self.steps_per_episode
        normalized_total_shares_bought = self.total_shares_bought / self.steps_per_episode
        normalized_total_shares_sold = self.total_shares_sold / self.steps_per_episode
        normalized_consecutive_trades = self.consecutive_trades / self.window_size
        normalized_time_since_last_trade = np.clip((self.current_step - self.last_trade_step) / max(self.min_trade_interval,1), 0, 10)
        # TODO experiment with log and sqrt
        normalized_consecutive_holds = np.log1p(self.consecutive_holds) / np.log1p(self.steps_per_episode - 1)
        normalized_last_portfolio_value = self.last_portfolio_value / self.initial_balance
        normalized_trade_history_length = len(self.trade_history) / self.steps_per_episode

        portfolio_info = np.array([
            normalized_portfolio_value,              # normalized portfolio value
            balance_ratio,                           # ratio of current and initial balance
            shares_value_ratio,                      # ratio of shares value to initial balance
            is_holding_shares,                       # boolean if shares are held
            position_pl_ratio,                       # profit/loss ratio on current position
            normalized_shares_held,
            normalized_total_trades,
            normalized_total_shares_bought,
            normalized_total_shares_sold,
            normalized_consecutive_trades,
            normalized_time_since_last_trade,
            normalized_consecutive_holds,
            normalized_last_portfolio_value,
            self.last_action if self.last_action is not None else 1,
            normalized_trade_history_length
        ], dtype=np.float32)

        state =  np.concatenate((normalized_features_flattened, portfolio_info)).astype(np.float32)
        
        # Track portfolio value history
        self.price_history.append(current_price)
        self.portfolio_values.append(portfolio_value)
            
        return state
    
    # TODO use next_state to add/subtract reward
    def calculate_reward(self, action, next_state, done):
        reward = 0
        current_price: float = self.data_nparray[self.current_step, self.close_prices_idx]
        current_portfolio_value = self.balance + self.shares_held * current_price
        portfolio_change = (current_portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value if self.last_portfolio_value != 0 else self.initial_balance)

        # profit/Loss based reward
        if self.shares_held > 0:
            current_profit_pct: float = (current_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0

            # update trailing stop price if profit increases
            if self.position_open and current_profit_pct > self.trailing_stop:
                self.trailing_stop_price = max(self.trailing_stop_price, current_price * (1 - self.trailing_stop))
            
            # TODO is this logic sound
            # trailing stop loss penalty
            if self.position_open and current_price < self.trailing_stop_price and self.trailing_stop_price > 0:
                reward -= abs(current_profit_pct) * self.loss_penalty_weight * 1.5

            # step-wise profit taking reward
            for level, weight in zip(self.profit_taking_levels, self.profit_taking_weights):
                if current_profit_pct > level:
                    reward += current_profit_pct * self.profit_reward_weight * weight
                    self.max_profit = max(self.max_profit, current_profit_pct)

            # loss limiting reward
            if current_profit_pct < self.loss_threshold:
                reward += current_profit_pct * self.loss_penalty_weight # this is a penalty since profit_pct is negative
                self.max_loss = min(self.max_loss, current_profit_pct)

                # maximum loss threshold penalty
                if current_profit_pct < self.max_loss_threshold:
                    reward += current_profit_pct * self.loss_penalty_weight * 2

            # trend following reward (reward for being in a profitable trade)
            if self.position_open and current_profit_pct > 0:
                reward += portfolio_change * self.trend_following_weight * self.shares_held * current_price / self.initial_balance # scale by position size
                last_trade = self.trade_history[-1]
                if last_trade['type'] == 'BUY' and current_price > last_trade['price']:
                    reward += portfolio_change * self.trend_following_weight
                elif last_trade['type'] == 'SELL' and current_price < last_trade['price']:
                    reward += portfolio_change * self.trend_following_weight
                    
        # reward for taking a valid trading action (buy or sell)
        if action in [0, 2] and not self._is_invalid_action(action):
            reward += self.trade_reward
            # additional reward for the very first trade to encourage exploration
            if len(self.trade_history) == 0 and self.current_step > self.window_size:
                reward += self.first_trade_reward
            # reward for switching action from previous valid action (discourage holding too long)
            if self.last_action in [0, 2] and action != self.last_action and not self._is_invalid_action(action):
                 reward += self.opposite_action_reward

        # penalty for holding
        if action == 1: # hold action
            self.consecutive_holds += 1
            # apply a increasing penalty for consecutive holds to discourage inaction
            hold_penalty = self.hold_penalty_base * (1 + 0.03 * self.consecutive_holds)
            hold_penalty = max(hold_penalty, self.hold_penalty_max) # cap the penalty
            reward += hold_penalty
        else:
            self.consecutive_holds = 0
            
        # penalty for invalid actions
        if self._is_invalid_action(action):
             reward -= self.invalid_action_penalty # significant penalty for trying to execute an invalid action

        # final episode reward based on overall profit/loss
        if done:
            final_profit = (current_portfolio_value - self.initial_balance) / self.initial_balance
            if final_profit > 0:
                reward += final_profit * self.profit_reward_weight * 3 # higher weight for final profit
            else:
                reward += final_profit * self.loss_penalty_weight * 2 # higher penalty for final loss

        self.last_portfolio_value = current_portfolio_value
        return reward
    
    def _is_invalid_action(self, action):
        """returns bool representing if the action is invalid"""
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]

        if action == 0:  # sell
            if self.shares_held <= 0:
                return True
            sell_amount = self.shares_held * current_price
            return sell_amount <= self.min_trade_amount
        elif action == 2:  # buy
            # cannot buy if no balance or already in a position
            # check if the amount to buy is below the minimum trade amount
            if self.balance <= 0 or self.position_open:
                 return True
            # calculate max shares that can be bought with current balance
            max_possible_shares = int(self.balance / (current_price * (1 + self.transaction_fee))) if current_price > 0 else 0
            # consider the maximum position size constraint
            max_allowed_shares = int(self.initial_balance * self.max_position_size / current_price) if current_price > 0 else 0
            shares_to_buy = min(max_possible_shares, max_allowed_shares)
            return shares_to_buy * current_price <= self.min_trade_amount
        return False
    
    def step(self, action):
        """
        Execute one step in the environment based on the agent's action
        Actions: 0=Sell all, 1=Hold, 2=Buy max
        """
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        reward = 0
        done = False
        trade_info = {}
        invalid_action = self._is_invalid_action(action)
        
        # check if minimum trading interval has passed since last trade
        can_trade = (self.current_step - self.last_trade_step) >= self.min_trade_interval

        # track the action
        self.action_history.append(action)

        if invalid_action:
            reward = self.calculate_reward(action, self._get_state(), False)
        else:
            # process valid actions
            if action == 0:  # sell
                if not can_trade:
                    reward -= 0.1
                    invalid_action = True
                else:
                    sell_amount = self.shares_held * current_price
                    fee = sell_amount * self.transaction_fee
                    trade_info = {
                        'type': 'SELL',
                        'shares': self.shares_held,
                        'price': current_price,
                        'amount': sell_amount,
                        'fee': fee,
                        'action': action
                    }

                    self.balance += (sell_amount - fee)
                    self.total_shares_sold += self.shares_held
                    self.shares_held = 0
                    self.last_trade_step = self.current_step
                    self.consecutive_trades += 1
                    self.entry_price = 0 
                    self.trailing_stop_price = 0 
                    self.position_open = False 
            elif action == 1:  # hold
                self.consecutive_trades = 0
                reward -= 0.0001
                # TODO combine logic below
                # check for stop-loss or trailing stop conditions during hold
                if self.position_open and self.shares_held > 0:
                    current_profit_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0
                    # stop Loss
                    if current_profit_pct < self.max_loss_threshold:
                        print(f"Stop Loss triggered at step {self.current_step}. Selling all shares.")
                        sell_amount = self.shares_held * current_price
                        fee = sell_amount * self.transaction_fee
                        trade_info = {
                            'type': 'STOP_LOSS_SELL',
                            'shares': self.shares_held,
                            'price': current_price,
                            'amount': sell_amount,
                            'fee': fee,
                            'action': 'FORCED_SELL'
                        }
                        self.balance += (sell_amount - fee)
                        self.total_shares_sold += sell_amount # Count this as a sale
                        self.shares_held = 0
                        self.entry_price = 0
                        self.trailing_stop_price = 0
                        self.position_open = False
                        reward -= self.stop_loss_penalty * 2
                        # this forced sell is not a result of the agent's action (1, Hold),
                        # might need to adjust how this affects the agent's learning.
                    # trailing stop loss
                    elif self.trailing_stop_price > 0 and current_price < self.trailing_stop_price:
                        print(f"Trailing Stop Loss triggered at step {self.current_step}. Selling all shares.")
                        sell_amount = self.shares_held * current_price
                        fee = sell_amount * self.transaction_fee
                        trade_info = {
                            'type': 'TRAILING_STOP_SELL',
                            'shares': self.shares_held,
                            'price': current_price,
                            'amount': sell_amount,
                            'fee': fee,
                            'action': 'FORCED_SELL'
                        }
                        self.balance += (sell_amount - fee)
                        self.total_shares_sold += sell_amount 
                        self.shares_held = 0
                        self.entry_price = 0
                        self.trailing_stop_price = 0
                        self.position_open = False
                        reward -= self.stop_loss_penalty
            elif action == 2:  # buy
                if not can_trade:
                    reward = -0.1
                    invalid_action = True
                else:
                    max_shares_possible = int(self.balance / (current_price * (1 + self.transaction_fee))) if current_price > 0 else 0
                    max_shares_allowed = int(self.initial_balance * self.max_position_size / current_price) if current_price > 0 else 0
                    shares_to_buy = min(max_shares_possible, max_shares_allowed)
                    buy_amount = shares_to_buy * current_price
                    fee = buy_amount * self.transaction_fee
                    cost = buy_amount + fee

                    trade_info = {
                        'type': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'amount': buy_amount,
                        'fee': fee,
                        'action': action
                    }

                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.entry_price = current_price # Set entry price on buy
                    self.trailing_stop_price = current_price * (1 - self.trailing_stop) # Set initial trailing stop
                    self.position_open = True # Position is open
                    self.total_shares_bought += shares_to_buy
                    self.last_trade_step = self.current_step
                    self.consecutive_trades += 1
                    self.total_cost += cost

        self.current_step += 1

        current_portfolio_value = self.balance + self.shares_held * current_price
        is_critical_loss = current_portfolio_value < self.initial_balance * 0.5
        # force sell all shares at the end of the episode or if critical loss
        if self.current_step >= len(self.data) - 1 or is_critical_loss:
            done = True
            if self.shares_held > 0:
                sell_amount = self.shares_held * current_price
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)
                self.shares_held = 0
            
            if is_critical_loss:
                reward -= 1
                  
        if done and self.mode == 'test':
            max_drawdown = self._calculate_max_drawdown(self.portfolio_values)
            sharpe_ratio = self._calculate_sharpe_ratio(self.portfolio_values)
            trade_info['max_drawdown'] = max_drawdown
            trade_info['sharpe_ratio'] = sharpe_ratio
            trade_info['portfolio_values'] = self.portfolio_values
            trade_info['price_history'] = self.price_history
            trade_info['action_history'] = self.action_history
            
            final_portfolio_value = self.balance
            return_rate = (final_portfolio_value - self.initial_balance) / self.initial_balance
            
            trade_info['final_balance'] = final_portfolio_value
            trade_info['return_rate'] = return_rate

        # append trade info if a trade occurred
        if trade_info:
            self.trade_history.append(trade_info)

        # only calculate if not an invalid action that was penalized immediately
        if not (invalid_action and reward < -0.0001):
            reward += self.calculate_reward(action, self._get_state(), done)

        self.last_action = action
        next_state = self._get_state()

        return next_state, reward, done, {
            'trade_info': trade_info,
            'invalid_action': invalid_action,
            'current_portfolio_value': current_portfolio_value,
            'current_price': current_price
        }
    
    def _calculate_max_drawdown(self, portfolio_values) -> np.float64:
        """
        Calculate the maximum drawdown from peak to trough
        """
        # convert to numpy array if not already
        values = np.array(portfolio_values)
        # calculate the running maximum
        running_max = np.maximum.accumulate(values)
        # calculate drawdown in percentage terms
        drawdown = (running_max - values) / running_max
        # get the maximum drawdown
        max_drawdown = np.max(drawdown)
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.02/252) -> np.float64:
        """
        Calculate the Sharpe ratio of the portfolio
        """
        # convert to numpy array if not already
        values = np.array(portfolio_values)
        # calculate daily returns
        daily_returns = np.diff(values) / values[:-1]
        # calculate excess returns over risk-free rate
        excess_returns = daily_returns - risk_free_rate
        # calculate Sharpe ratio (annualized)
        if np.std(excess_returns) == 0:
            return 0
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio

    @property
    def state_size(self) -> int:
        """
        Calculate state size dynamically based on features and portfolio info
        """
        if not hasattr(self, '_state_size_cached'):
            # number of features plus volatility and price_change
            num_data_features = self.data.columns.size
            # flattened normalized features
            features_part_len = self.window_size * num_data_features
            # portfolio info part (portfolio_info's length)
            portfolio_part_len = 15
            self._state_size_cached = features_part_len + portfolio_part_len
        return self._state_size_cached


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        # data features + portfolio_info up until the last 6
        self.state_layers = nn.Sequential(
            nn.Linear(state_size - 6, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # the afromentioned last 6
        self.constraint_layers = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # action layer
        self.action_layer = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, action_size)
        )
        
        # initialize optimized weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        state_features = self.state_layers(x[:, :-6]) # data features + portfolio_info up until the last 6
        constraint_features = self.constraint_layers(x[:, -6:])
        combined_features = torch.cat([state_features, constraint_features], dim=1)
        
        return self.action_layer(combined_features)


class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQNNetwork, self).__init__()

        self.action_size = action_size
        # data features + portfolio_info up until the last 6
        self.state_layers = nn.Sequential(
            nn.Linear(state_size - 6, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # the afromentioned last 6
        self.constraint_layers = nn.Sequential(
            nn.Linear(6, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # --- Dueling DQN specific layers ---

        # The combined features dimension is 64 (from state_layers) + 16 (from constraint_layers) = 80

        # Value stream: Estimates the state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(80, 64), # Input size matches combined_features
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # Output a single scalar for the value
        )

        # Advantage stream: Estimates the advantage A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(80, 64), # Input size matches combined_features
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, action_size) # Output a value for each action
        )

        # Initialize weights using the same method
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights using Kaiming Normal and biases to zero.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Dueling DQN network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The calculated Q-values for each action.
        """
        # Process state features and constraint features separately
        # Assuming the last 6 features are the constraints
        state_features = self.state_layers(x[:, :-6])
        constraint_features = self.constraint_layers(x[:, -6:]) # Slicing the last 6 features

        # Concatenate the outputs from both streams
        combined_features = torch.cat([state_features, constraint_features], dim=1)

        # Pass combined features through Value and Advantage streams
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)

        # Combine Value and Advantage to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay for more efficient learning
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = beta  # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # Beta increases over time for more correction
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
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get samples and calculate importance sampling weights
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()  # Normalize weights
        
        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(map(list, zip(*samples)))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        
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


class DQNAgent:
    """
    Deep Q-Network agent with prioritized experience replay and dueling architecture option
    """
    def __init__(
        self, 
        state_size: int,
        action_size: int,
        total_steps: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        decay_rate_multiplier: float = 0.995,
        epsilon_min: float = 0.01,
        epsilon_decay_target_pct: float=1,
        batch_size: int = 256,
        memory_size: int = 100000,
        update_frequency: int = 4,
        target_update_frequency: int = 100,
        use_dueling: bool = True,
        use_prioritized: bool = True
    ):
        self.state_size: int = state_size
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
        self.current_step = 0

        # early forced exploration settings
        self.initial_exploration_episodes = 20  # 초기 탐색 에피소드 증가
        self.force_trade_probability = 0.8  # 강제 거래 확률 증가
        self.min_epsilon = 0.1
        self.current_episode = 0

        # device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # network initialization
        if use_dueling:
            self.main_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
        else:
            self.main_network = DQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size).to(self.device)
            
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() 

        # optimizer
        self.optimizer = optim.AdamW(
            self.main_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5, 
            amsgrad=True
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=3,
            verbose=True
        )
        
        # Memory setup
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
            
        self.loss_fn = nn.SmoothL1Loss()

        # Training parameters
        self.update_counter = 0
        self.target_update_frequency = target_update_frequency
        self.update_frequency = update_frequency
        self.training_steps = 0
        
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
            loss = (is_weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()
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
            self.epsilon = (self.epsilon_min) ** ((self.current_step / (self.total_steps * self.epsilon_decay_target_pct)) ** self.decay_rate_multiplier)

    def load(self, file_path):
        """Load model weights from file"""
        self.main_network.load_state_dict(torch.load(file_path, map_location=self.device))
        self.target_network.load_state_dict(self.main_network.state_dict())
        print(f"Model loaded from {file_path}")

    def save(self, file_path):
        """Save model weights to file"""
        torch.save(self.main_network.state_dict(), file_path)
        print(f"Model saved to {file_path}")


def train_agent(env: StockTradingEnv, 
                agent: DQNAgent, 
                episodes: int=100, 
                validation_env: Optional[StockTradingEnv]=None,
                validation_frequency: int=5,
                early_stopping_patience: int=10):
    """
    Train the agent with optional validation and early stopping
    """
    scores = []
    balances = []
    validation_scores = []
    invalid_action_counts = []
    
    best_validation_score = float('-inf')
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    
    for e in range(episodes):
        agent.current_episode = e
        state = env.reset()
        score = 0
        done = False
        invalid_actions = 0
        trades_this_episode = 0

        while not done:
            agent.current_step += 1
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if info.get('invalid_action'):
                invalid_actions += 1
            if info.get('trade_info'):
                trades_this_episode += 1
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            score += reward

        agent.scheduler.step(score)
        scores.append(score)
        balances.append(env.balance)
        invalid_action_counts.append(invalid_actions)
        
        if (e + 1) % 5 == 0:
            print(f"Episode: {e+1}/{episodes}, "
                  f"Score: {score:.4f}, "
                  f"Balance: {env.balance:.2f}, "
                  f"Trades: {trades_this_episode}, "
                  f"Invalid Actions: {invalid_actions}, "
                  f"Epsilon: {agent.epsilon:.4f}")
        
        # validation if provided
        if validation_env is not None and (e + 1) % validation_frequency == 0:
            validation_score = evaluate_agent(validation_env, agent, episodes=1, verbose=False)
            validation_scores.append(validation_score)
            
            # early stopping logic
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                patience_counter = 0
                # save best model state
                best_model_state = {k: v.cpu() for k, v in agent.main_network.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at episode {e+1}. Best validation score: {best_validation_score:.4f}")
                # restore best model
                if best_model_state:
                    agent.main_network.load_state_dict(best_model_state)
                    agent.target_network.load_state_dict(best_model_state)
                break
            
    print(f'Training Time: {format_duration(time.time() - start_time)}')
    
    training_metrics = {
        'scores': scores,
        'balances': balances,
        'validation_scores': validation_scores,
        'loss_history': agent.loss_history,
        'avg_q_values': agent.avg_q_values,
        'invalid_action_count': invalid_action_counts
    }
    
    return training_metrics


def evaluate_agent(env: StockTradingEnv, agent: DQNAgent, episodes: int=10, verbose: bool=True):
    """
    Evaluate the agent's performance
    """
    total_return_rate = 0
    total_trades = 0
    total_invalid_actions = 0
    start_time = time.time()
    
    for e in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_trades = 0
        episode_invalid_actions = 0
        episode_rewards = []

        while not done:
            action = agent.act(state, training=False)  # evaluation mode
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_rewards.append(reward)
            episode_reward += reward
            
            if info.get('invalid_action'):
                episode_invalid_actions += 1
            if info.get('trade_info'):
                episode_trades += 1
            
        # get return rate for this episode
        final_portfolio_value = env.balance + env.shares_held * env.data_nparray[env.current_step, env.close_prices_idx]
        return_rate = (final_portfolio_value - env.initial_balance) / env.initial_balance
        total_return_rate += return_rate
        total_trades += episode_trades
        total_invalid_actions += episode_invalid_actions
        
        if verbose:
            print(f"\nEvaluation Episode {e+1}/{episodes}")
            print(f"Return Rate: {return_rate:.4f}")
            print(f"Final Balance: {env.balance:.2f}")
            print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
            print(f"Trades: {episode_trades}")
            print(f"Invalid Actions: {episode_invalid_actions}")
            print(f"Average Reward: {np.mean(episode_rewards):.4f}")
            print(f"Total Shares Bought: {env.total_shares_bought}")
            print(f"Total Shares Sold: {env.total_shares_sold}")
            print(f"Episode Score: {episode_reward:.4f}")
            
    
    avg_return_rate = total_return_rate / episodes
    if verbose:
        print(f"\nEvaluation Summary:")
        print(f"Average Return: {(total_return_rate / episodes):.4f}")
        print(f"Average Trades per Episode: {(total_trades / episodes):.1f}")
        print(f"Average Invalid Actions per Episode: {(total_invalid_actions / episodes):.1f}")
        print(f'Evaluation Time: {format_duration(time.time() - start_time)}')

    return (avg_return_rate if episodes > 1 else return_rate)


def plot_training_results(metrics, model_name="DQN", validation_frequency: int=5,):
    """
    Visualize training and performance metrics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training scores
    plt.subplot(2, 2, 1)
    plt.plot(metrics['scores'], label='Training Score')
    if metrics['validation_scores']:
        # Plot validation scores at their corresponding episodes
        validation_episodes = [i*validation_frequency for i in range(len(metrics['validation_scores']))]
        plt.plot(validation_episodes, metrics['validation_scores'], 'r-', label='Validation Score')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Score')
    plt.title(f'{model_name} Learning Curve - Cumulative Score')
    plt.legend()
    
    # Plot final balance after each episode
    plt.subplot(2, 2, 2)
    plt.plot(metrics['balances'])
    plt.xlabel('Episode')
    plt.ylabel('Final Balance ($)')
    plt.title('Portfolio Value at End of Episode')
    
    # Plot loss history
    if metrics['loss_history']:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['loss_history'])
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
    
    # Plot average Q-values
    if metrics['avg_q_values']:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['avg_q_values'])
        plt.xlabel('Action Selection')
        plt.ylabel('Average Q-Value')
        plt.title('Average Q-Values During Training')
    
    plt.tight_layout()
    return plt

def plot_backtest_results(portfolio_values, price_history, action_history, ticker: str):
    """
    Visualize backtesting results including price chart, portfolio value,
    and buy/sell actions
    """
    plt.figure(figsize=(15, 10))
    
    # Plot stock price
    plt.subplot(2, 1, 1)
    plt.plot(price_history, label=f'{ticker} Price')
    
    # Mark buy and sell actions
    buy_indices = [i for i, a in enumerate(action_history) if a == 2]
    sell_indices = [i for i, a in enumerate(action_history) if a == 0]
    
    if buy_indices:
        plt.scatter(buy_indices, [price_history[i] for i in buy_indices], 
                   color='green', marker='^', s=100, label='Buy')
    if sell_indices:
        plt.scatter(sell_indices, [price_history[i] for i in sell_indices], 
                   color='red', marker='v', s=100, label='Sell')
    
    plt.xlabel('Trading Step')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} Price and Trading Actions')
    plt.legend()
    
    # Plot portfolio value
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values, label='Portfolio Value')
    
    # Calculate and plot buy-and-hold strategy for comparison
    initial_balance = portfolio_values[0]
    initial_price = price_history[0]
    shares_bought = initial_balance / initial_price
    buy_hold_values = [shares_bought * price for price in price_history]
    plt.plot(buy_hold_values, '--', label='Buy & Hold Strategy')
    
    plt.xlabel('Trading Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Comparison')
    plt.legend()
    
    plt.tight_layout()
    return plt

def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock data
    """
    drop_cols = ['timestamp']
    file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    
    return df

def load_stock_data(ticker: str) -> pd.DataFrame:
    drop_cols = ['timestamp']
    file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cutoff = pd.Timestamp('2025-04-29 08:00:00', tz='UTC')

    df = df[df['timestamp'] >= cutoff]
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    
    return df

def split_data(data: pd.DataFrame, train_ratio=TRAIN_RATIO, val_ratio=VALID_RATIO):
    """
    Split data chronologically into train, validation, and test sets
    """
    # Calculate split indices
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    
    # Split data
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    val_data = data.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_data = data.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"Data split: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")
    
    return train_data, val_data, test_data

def main():
    print("Let's get this bread")
    # parameters
    ticker = 'AAPL'
    window_size = 60 
    initial_balance = 10000
    transaction_fee = 0.001 
    
    # load and prepare data
    data = load_stock_data(ticker)
    train_data, val_data, test_data = split_data(data)
    
    # create environments
    train_env = StockTradingEnv(
        train_data, 
        initial_balance=initial_balance, 
        transaction_fee=transaction_fee, 
        window_size=window_size,
        mode='train'
    )
    
    val_env = StockTradingEnv(
        val_data, 
        initial_balance=initial_balance, 
        transaction_fee=transaction_fee, 
        window_size=window_size,
        mode='validation'
    )
    
    test_env = StockTradingEnv(
        test_data, 
        initial_balance=initial_balance, 
        transaction_fee=transaction_fee, 
        window_size=window_size,
        mode='test'
    )
    
    # define agent
    episodes = 150
    steps_per_episode = len(train_data) - window_size
    state_size = train_env.state_size
    action_size = 3  # sell(0), hold(1), buy(2)
    epsilon = 1.0
    epsilon_min = 0.05
    total_steps = steps_per_episode * episodes
    decay_rate_multiplier = 1
    
    # initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        total_steps=total_steps,
        learning_rate=0.0005,  # Lower learning rate for stability
        discount_factor=0.97,  # Higher discount factor for longer-term rewards
        epsilon=epsilon,
        decay_rate_multiplier=decay_rate_multiplier,
        epsilon_min=epsilon_min,
        epsilon_decay_target_pct=1,
        batch_size=1024,       # Larger batch size for more stable learning
        memory_size=20000,    # Larger memory for better experience diversity
        update_frequency=4,   # Update every 4 steps for efficiency
        target_update_frequency=200,  # Less frequent target updates for stability
        use_dueling=True,     # Use dueling architecture
        use_prioritized=True  # Use prioritized replay
    )
    
    # train agent with validation-based early stopping
    print("Starting training...")
    training_metrics = train_agent(
        train_env, 
        agent, 
        episodes=episodes,
        validation_env=val_env,
        early_stopping_patience=15
    )
    
    # plot training results
    training_plot = plot_training_results(training_metrics)
    training_plot.savefig(f'dqn_{ticker}_training_results.png')
    
    # evaluate on test set
    print("\nEvaluating on test set...")
    avg_return = evaluate_agent(test_env, agent, episodes=1)
    
    state = test_env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
    
    info = info['trade_info']
    
    # Plot backtest results
    if 'portfolio_values' in info:
        backtest_plot = plot_backtest_results(
            info['portfolio_values'], 
            info['price_history'], 
            info['action_history'], 
            ticker=ticker
        )
        backtest_plot.savefig(f"dqn_{ticker}_{info['return_rate']*100:.2f}_backtest_results.png")
        
    # Save model
    model_path = f"{MODELS_DIR}/dqn_{ticker}_{info['return_rate']*100:.2f}_model.pth"
    agent.save(model_path)
    
    # Print final metrics
    print(f"\nTest Results for {ticker}:")
    print(f"Final Balance: ${info['final_balance']:.2f}")
    print(f"Return Rate: {info['return_rate']:.4f} ({info['return_rate']*100:.2f}%)")
    print(f"Max Drawdown: {info['max_drawdown']:.4f} ({info['max_drawdown']*100:.2f}%)")
    print(f"Sharpe Ratio: {info['sharpe_ratio']:.4f}")


if __name__ == '__main__':
    main()