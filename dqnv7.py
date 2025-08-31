import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from sklearn.preprocessing import StandardScaler
from torch.types import Number
from src.config.config import DATA_DIR 

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class StockTradingEnv:
    def __init__(self, data, initial_balance=50000, transaction_fee=0.00005):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        # 거래 조건 설정
        self.min_trade_interval = 2  # 거래 간격 더 감소
        self.max_position_size = 0.7  # 최대 포지션 크기 증가
        self.min_trade_amount = 300   # 최소 거래 금액 더 감소
        self.max_consecutive_trades = 40  # 연속 거래 제한 증가

        # 보상 관련 설정
        self.hold_penalty_base = -0.02  # 보유 패널티 더 감소
        self.hold_penalty_max = -0.15   # 최대 보유 패널티 더 감소
        self.profit_reward_weight = 50  # 수익 보상 가중치 증가
        self.loss_penalty_weight = 25   # 손실 패널티 가중치 증가
        self.trade_reward = 0.2         # 거래 실행 보상 더 감소
        self.first_trade_reward = 0.1   # 첫 거래 보상 더 감소
        self.opposite_action_reward = 0.1  # 반대 행동 보상 더 감소

        # 수익/손실 임계값 설정
        self.profit_threshold = 0.01    # 수익 실현 임계값 더 감소
        self.loss_threshold = -0.003    # 손실 제한 임계값 더 감소
        self.trailing_stop = 0.008      # 트레일링 스탑 감소
        self.max_profit_threshold = 0.02  # 최대 수익 실현 임계값 감소
        self.max_loss_threshold = -0.008  # 최대 손실 제한 임계값 감소

        # 추가 수익성 관련 설정
        self.profit_taking_levels = [0.01, 0.02, 0.03]  # 단계별 수익 실현
        self.profit_taking_weights = [1.0, 1.5, 2.0]    # 단계별 보상 가중치
        self.volatility_threshold = 0.02  # 변동성 임계값
        self.trend_following_weight = 1.2  # 추세 추종 가중치

        # 상태 변수 초기화
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.total_buys = 0
        self.total_sales = 0
        self.consecutive_trades = 0
        self.consecutive_holds = 0
        self.last_portfolio_value = initial_balance
        self.last_action = None
        self.last_trade_step = -1
        self.trade_history = []
        self.entry_price = 0
        self.max_profit = 0
        self.max_loss = 0
        self.trailing_stop_price = 0  # 트레일링 스탑 가격
        self.position_open = False    # 포지션 상태

        # 데이터 변환
        self.price_data = data.loc[:,:'vwap'].values
        self.technical_indicators = data.loc[:,'stochrsi_k_14_1min':].values

        # 스케일러 초기화
        self.scaler = StandardScaler()
        # Fit the scaler on a representative portion of data
        initial_features = self._get_features(slice(self.max_consecutive_trades, len(data)))
        # Flatten the features for scaler fitting
        flattened_features = initial_features.reshape(-1, initial_features.shape[-1])
        self.scaler.fit(flattened_features)

        # Initial state setting
        self.reset()

    def reset(self):
        self.current_step = self.max_consecutive_trades # Start after the lookback window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.total_buys = 0
        self.total_sales = 0
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
        return self._get_state()

    def _get_features(self, index):
        """기본 특성 추출 (가격, 거래량, 지표)"""
        # Ensure index handles slicing and single step
        if isinstance(index, int):
             # Adjust index to get the lookback window
             start_idx = max(0, index - self.max_consecutive_trades + 1)
             end_idx = index + 1
        elif isinstance(index, slice):
             start_idx = max(0, index.start - self.max_consecutive_trades + 1) if index.start is not None else 0
             end_idx = index.stop if index.stop is not None else len(self.data)
        else:
             raise TypeError(f"Unsupported index type: {type(index)}")

        # Select data within the lookback window
        price_data_window = self.price_data[start_idx:end_idx]
        tech_ind_window = self.technical_indicators[start_idx:end_idx]


        # Combine price and technical indicators
        features = np.column_stack((
            price_data_window[:, 3],  # close price
            price_data_window[:, 4],  # volume
            tech_ind_window[:, 0],    # MA5
            tech_ind_window[:, 1],    # MA10
            tech_ind_window[:, 2],    # MA20
            tech_ind_window[:, 3],    # RSI
            tech_ind_window[:, 4],    # MACD
            tech_ind_window[:, 5]     # Signal
        ))

        return features

    def _get_state(self):
        """상태 벡터 생성"""
        features = self._get_features(self.current_step)
        normalized_features = self.scaler.transform(features)
        current_price = self.price_data[self.current_step, 3] # Use close price for current price

        # Calculate portfolio value and performance metrics
        portfolio_value = self.balance + self.shares_held * current_price
        # Avoid division by zero and handle initial state
        portfolio_change = (portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value if self.last_portfolio_value != 0 else self.initial_balance)
        portfolio_change = np.clip(portfolio_change, -1, 1) # Clip large changes

        # Portfolio information vector (11 elements)
        portfolio_info = np.array([
            self.balance / self.initial_balance, # Normalized balance
            self.shares_held, # Current shares held
            self.total_trades, # Total trades
            self.total_buys, # Total buys
            self.total_sales, # Total sales
            self.consecutive_trades, # Consecutive trades
            self.current_step - self.last_trade_step if self.last_trade_step != -1 else 0, # Steps since last trade
            self.consecutive_holds, # Consecutive holds
            portfolio_value / self.initial_balance, # Normalized portfolio value
            self.last_action if self.last_action is not None else 1, # Last action (0: Sell, 1: Hold, 2: Buy)
            len(self.trade_history) # Number of trades in history
        ], dtype=np.float32)

        # Normalize portfolio info features that have a clear maximum or can be reasonably scaled
        # Shares held can be normalized by max position size
        portfolio_info[1] = portfolio_info[1] / self.max_position_size if self.max_position_size > 0 else 0 # Shares held normalized

        # Normalize metrics that grow over time by a large number or episode step
        portfolio_info[2] = portfolio_info[2] / 1000 # Total trades
        portfolio_info[3] = portfolio_info[3] / 1000 # Total buys
        portfolio_info[4] = portfolio_info[4] / 1000 # Total sales
        portfolio_info[5] = portfolio_info[5] / self.max_consecutive_trades # Consecutive trades
        portfolio_info[6] = np.clip(portfolio_info[6] / 100, 0, 10) # Steps since last trade, clipped
        portfolio_info[7] = portfolio_info[7] / 100 # Consecutive holds
        portfolio_info[10] = portfolio_info[10] / 1000 # Trade history length


        state = np.concatenate((normalized_features.flatten(), portfolio_info)).astype(np.float32)

        if self.current_step == self.max_consecutive_trades:
             print(f"\nState vector composition:")
             print(f"Normalized features shape: {normalized_features.flatten().shape}")
             print(f"Portfolio info shape: {portfolio_info.shape}")
             print(f"Total state vector size: {state.shape}")


        return state

    def calculate_reward(self, action, next_state, done):
        reward = 0
        current_price = self.price_data[self.current_step, 3] # Use close price
        current_portfolio_value = self.balance + self.shares_held * current_price
        portfolio_change = (current_portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value if self.last_portfolio_value != 0 else self.initial_balance)


        # Profit/Loss based reward
        if self.shares_held > 0:
            current_profit_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0

            # Trailing stop loss logic
            if self.position_open and current_price < self.trailing_stop_price and self.trailing_stop_price > 0:
                # This condition should ideally trigger a sell action in the step method, but adding a reward component here
                # encourages the agent to learn to avoid such situations or take action.
                reward -= abs(current_profit_pct) * self.loss_penalty_weight * 1.5 # Increased penalty

            # Step-wise profit taking reward
            for level, weight in zip(self.profit_taking_levels, self.profit_taking_weights):
                if current_profit_pct > level:
                    reward += current_profit_pct * self.profit_reward_weight * weight
                    self.max_profit = max(self.max_profit, current_profit_pct)

            # Loss limiting reward (can be negative if loss exceeds threshold)
            if current_profit_pct < self.loss_threshold:
                reward += current_profit_pct * self.loss_penalty_weight # This is a penalty since profit_pct is negative
                self.max_loss = min(self.max_loss, current_profit_pct)

                # Maximum loss threshold penalty
                if current_profit_pct < self.max_loss_threshold:
                    reward += current_profit_pct * self.loss_penalty_weight * 2 # Increased penalty

            # Update trailing stop price if profit increases
            if self.position_open and current_profit_pct > self.trailing_stop:
                self.trailing_stop_price = max(self.trailing_stop_price, current_price * (1 - self.trailing_stop))


            # Trend following reward (reward for being in a profitable trade)
            if self.position_open and current_profit_pct > 0:
                 reward += portfolio_change * self.trend_following_weight * self.shares_held * current_price / self.initial_balance # Scale by position size


        # Reward for taking a valid trading action (Buy or Sell)
        if action in [0, 2] and not self._is_invalid_action(action):
            reward += self.trade_reward
            # Additional reward for the very first trade to encourage exploration
            if len(self.trade_history) == 0 and self.current_step > self.max_consecutive_trades:
                reward += self.first_trade_reward
            # Reward for switching action from previous valid action (discourage holding too long)
            if self.last_action in [0, 2] and action != self.last_action and not self._is_invalid_action(action):
                 reward += self.opposite_action_reward


        # Penalty for holding
        if action == 1: # Hold action
            self.consecutive_holds += 1
            # Apply a increasing penalty for consecutive holds to discourage inaction
            hold_penalty = self.hold_penalty_base * (1 + 0.03 * self.consecutive_holds)
            hold_penalty = max(hold_penalty, self.hold_penalty_max) # Cap the penalty
            reward += hold_penalty
        else:
            self.consecutive_holds = 0 # Reset consecutive holds on a trade action

        # Penalty for invalid actions
        if self._is_invalid_action(action):
             reward -= 0.5 # Significant penalty for trying to execute an invalid action


        # Final episode reward based on overall profit/loss
        if done:
            final_profit = (current_portfolio_value - self.initial_balance) / self.initial_balance
            if final_profit > 0:
                reward += final_profit * self.profit_reward_weight * 3 # Higher weight for final profit
            else:
                reward += final_profit * self.loss_penalty_weight * 2 # Higher penalty for final loss


        self.last_portfolio_value = current_portfolio_value
        self.last_action = action # Update last action
        return reward

    def _is_invalid_action(self, action):
        """유효하지 않은 액션인지 확인"""
        current_price = self.price_data[self.current_step, 3] # Use close price

        if action == 0:  # 매도 (Sell)
            # Cannot sell if no shares held
            return self.shares_held <= 0
        elif action == 2:  # 매수 (Buy)
            # Cannot buy if no balance or already in a position
            # Also check if the amount to buy is below the minimum trade amount
            if self.balance <= 0 or self.position_open:
                 return True
            # Estimate maximum shares that can be bought with current balance
            max_possible_shares = int(self.balance / (current_price * (1 + self.transaction_fee))) if current_price > 0 else 0
            # Consider the maximum position size constraint
            max_allowed_shares = int(self.initial_balance * self.max_position_size / current_price) if current_price > 0 else 0
            shares_to_buy = min(max_possible_shares, max_allowed_shares)
            return shares_to_buy * current_price < self.min_trade_amount
        # Action 1 (Hold) is always valid if within data range
        return False

    def step(self, action):
        current_price = self.price_data[self.current_step, 3] # Use close price
        reward = 0
        done = False
        trade_info = None
        invalid_action = self._is_invalid_action(action) # Check for invalid action at the beginning of the step

        # Apply penalty immediately if the action is invalid
        if invalid_action:
             reward = self.calculate_reward(action, self._get_state(), False) # Calculate penalty
        else:
            # Process valid actions
            if action == 0:  # 매도 (Sell)
                # This block is only executed if action is valid (shares_held > 0)
                sell_amount = self.shares_held * current_price
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)

                self.total_sales += sell_amount

                trade_info = {
                    'type': 'SELL',
                    'shares': self.shares_held,
                    'price': current_price,
                    'amount': sell_amount,
                    'fee': fee,
                    'action': action,
                    'step': self.current_step
                }

                self.shares_held = 0
                self.last_trade_step = self.current_step
                self.consecutive_trades += 1
                self.entry_price = 0 # Reset entry price on sell
                self.trailing_stop_price = 0 # Reset trailing stop on sell
                self.position_open = False # Position is closed

            elif action == 1:  # 보유 (Hold)
                self.consecutive_trades = 0 # Reset consecutive trades on hold

                # Check for stop-loss or trailing stop conditions during hold
                if self.position_open and self.shares_held > 0:
                    current_profit_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0

                    # Stop Loss
                    if current_profit_pct < self.max_loss_threshold:
                        print(f"Stop Loss triggered at step {self.current_step}. Selling all shares.")
                        # Execute a forced sell
                        sell_amount = self.shares_held * current_price
                        fee = sell_amount * self.transaction_fee
                        self.balance += (sell_amount - fee)
                        self.total_sales += sell_amount # Count this as a sale
                        self.shares_held = 0
                        self.entry_price = 0
                        self.trailing_stop_price = 0
                        self.position_open = False
                        reward -= 0.5 # Penalty for hitting stop loss (can be adjusted in calculate_reward)
                        # Note: This forced sell is not a result of the agent's action (1, Hold),
                        # it's an environment rule. We might need to adjust how this affects the agent's learning.
                        # For now, the penalty is added to the reward of the 'Hold' action.
                        trade_info = {
                            'type': 'STOP_LOSS_SELL',
                            'shares': self.shares_held, # Shares before selling
                            'price': current_price,
                            'amount': sell_amount,
                            'fee': fee,
                            'action': 'FORCED_SELL',
                            'step': self.current_step
                        }

                    # Trailing Stop Loss
                    elif self.position_open and self.shares_held > 0 and self.trailing_stop_price > 0 and current_price < self.trailing_stop_price:
                        print(f"Trailing Stop Loss triggered at step {self.current_step}. Selling all shares.")
                         # Execute a forced sell
                        sell_amount = self.shares_held * current_price
                        fee = sell_amount * self.transaction_fee
                        self.balance += (sell_amount - fee)
                        self.total_sales += sell_amount # Count this as a sale
                        self.shares_held = 0
                        self.entry_price = 0
                        self.trailing_stop_price = 0
                        self.position_open = False
                        reward -= 0.3 # Smaller penalty than stop loss
                        trade_info = {
                            'type': 'TRAILING_STOP_SELL',
                            'shares': self.shares_held, # Shares before selling
                            'price': current_price,
                            'amount': sell_amount,
                            'fee': fee,
                            'action': 'FORCED_SELL',
                            'step': self.current_step
                        }


            elif action == 2:  # 매수 (Buy)
                # This block is only executed if action is valid (balance > 0, not in position, buy amount >= min_trade_amount)
                max_shares_possible = int(self.balance / (current_price * (1 + self.transaction_fee))) if current_price > 0 else 0
                max_shares_allowed = int(self.initial_balance * self.max_position_size / current_price) if current_price > 0 else 0
                shares_to_buy = min(max_shares_possible, max_shares_allowed)

                buy_amount = shares_to_buy * current_price
                fee = buy_amount * self.transaction_fee
                cost = buy_amount + fee

                self.balance -= cost
                self.shares_held += shares_to_buy
                self.entry_price = current_price # Set entry price on buy
                self.trailing_stop_price = current_price * (1 - self.trailing_stop) # Set initial trailing stop
                self.position_open = True # Position is open

                self.total_buys += shares_to_buy

                trade_info = {
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'amount': buy_amount,
                    'fee': fee,
                    'action': action,
                    'step': self.current_step
                }

                self.last_trade_step = self.current_step
                self.consecutive_trades += 1

        # Increment step regardless of action validity
        self.current_step += 1

        # Check for end of data or critical loss
        if self.current_step >= len(self.price_data):
            done = True
            # Force sell all shares at the end of the episode
            if self.shares_held > 0:
                sell_amount = self.shares_held * self.price_data[self.current_step -1, 3] # Use last available price
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)
                self.shares_held = 0
                print(f"End of episode. Forced sell remaining shares at step {self.current_step -1}.")


        # Check for critical loss leading to early termination
        current_portfolio_value = self.balance + self.shares_held * current_price if self.current_step < len(self.price_data) else self.balance # Use current price if available, else final balance
        if current_portfolio_value < self.initial_balance * 0.5: # 50% loss threshold
             print(f"Critical loss triggered at step {self.current_step}. Portfolio value below 50% of initial balance. Terminating episode.")
             done = True
             # Force sell remaining shares before terminating
             if self.shares_held > 0 and self.current_step < len(self.price_data):
                sell_amount = self.shares_held * self.price_data[self.current_step, 3]
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)
                self.shares_held = 0
             elif self.shares_held > 0 and self.current_step >= len(self.price_data):
                  sell_amount = self.shares_held * self.price_data[self.current_step-1, 3]
                  fee = sell_amount * self.transaction_fee
                  self.balance += (sell_amount - fee)
                  self.shares_held = 0


        # Append trade info if a trade occurred
        if trade_info:
            self.trade_history.append(trade_info)


        # Calculate reward for the executed action (or penalty for invalid)
        # Only calculate if not an invalid action that was penalized immediately
        if not (invalid_action and reward < 0): # Avoid double penalizing
             reward = self.calculate_reward(action, self._get_state(), done)


        next_state = self._get_state()


        return next_state, reward, done, {
            'trade_info': trade_info,
            'invalid_action': invalid_action,
            'current_portfolio_value': current_portfolio_value,
            'current_price': current_price
        }

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture that separates state value and advantage functions
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQNNetwork, self).__init__()
        # Common feature layer
        print('state_size:', state_size)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        # Check if input is a single sample and add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))
    
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        # 기존 상태 처리 레이어 (더 깊게 수정)
        self.state_layers = nn.Sequential(
            nn.Linear(state_size - 5, 1024),  # 레이어 크기 증가
            nn.ReLU(),
            nn.BatchNorm1d(1024),  # 배치 정규화 추가
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
        
        # 제약조건 처리 레이어 (더 깊게 수정)
        self.constraint_layers = nn.Sequential(
            nn.Linear(5, 64),
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
        
        # 최종 액션 선택 레이어
        self.action_layer = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, action_size)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 상태 처리
        state_features = self.state_layers(x[:, :-5])
        
        # 제약조건 처리
        constraint_features = self.constraint_layers(x[:, -5:])
        
        # 특성 결합
        combined_features = torch.cat([state_features, constraint_features], dim=1)
        
        # 액션 선택
        return self.action_layer(combined_features)


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
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
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
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.learning_rate: float = learning_rate
        self.use_dueling: bool = use_dueling
        self.use_prioritized: bool = use_prioritized
        self.current_step = 0

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Network initialization
        if use_dueling:
            self.main_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
        else:
            self.main_network = DQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size).to(self.device)
            
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() 

        # Optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        
        # Memory setup
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)

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
        # Exploration during training
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        # Convert state to tensor
        state = torch.FloatTensor(state).to(self.device)
        
        # Get Q-values from network
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(state)
        self.main_network.train()
        
        # Track average Q-values during training
        if training:
            self.avg_q_values.append(q_values.mean().item())
            
        return torch.argmax(q_values, dim=1).item()

    def train(self):
        """
        Train the agent by sampling from replay buffer
        """
        # Skip if not enough samples
        if self.use_prioritized:
            if len(self.memory) < self.batch_size:
                return
        else:
            if len(self.memory) < self.batch_size:
                return
                
        self.training_steps += 1
        
        # Only update every update_frequency steps
        if self.training_steps % self.update_frequency != 0:
            return
            
        # Sample from memory
        if self.use_prioritized:
            batch, indices, is_weights = self.memory.sample(self.batch_size)
            if batch is None:  # Not enough samples
                return
                
            states, actions, rewards, next_states, dones = batch
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.array([experience[0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q-values
        q_values = self.main_network(states).gather(1, actions)

        # Double DQN: Get actions from main network
        with torch.no_grad():
            next_actions = self.main_network(next_states).max(1, keepdim=True)[1]
            # Get Q-values for those actions from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Calculate target Q-values
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # Calculate loss
        if self.use_prioritized:
            # TD errors for updating priorities
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            # Weighted MSE loss
            loss = (is_weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()
        else:
            loss = F.mse_loss(q_values, target_q_values)
            
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in buffer
        if self.use_prioritized:
            self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant for stability

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            
        # Track loss
        self.loss_history.append(loss.item())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = (self.epsilon_min) ** (self.current_step / self.total_steps)

    def load(self, name):
        """Load model weights from file"""
        self.main_network.load_state_dict(torch.load(name, map_location=self.device))
        self.target_network.load_state_dict(self.main_network.state_dict())
        print(f"Model loaded from {name}")

    def save(self, name):
        """Save model weights to file"""
        torch.save(self.main_network.state_dict(), name)
        print(f"Model saved to {name}")

def train_agent(env, agent, episodes=30):
    scores = []
    balances = []
    invalid_action_counts = []
    start_time = time.time()

    max_steps_per_episode = 2000
    no_change_count = 0
    last_portfolio_value_check = env.initial_balance
    no_change_threshold = 200  # Number of steps with minimal portfolio change to trigger check
    min_steps_before_check = 500  # Minimum steps before checking for no change

    for e in range(episodes):
        agent.current_episode = e
        episode_start = time.time()
        state = env.reset()
        score = 0
        done = False
        step_count = 0
        invalid_actions = 0
        trades_this_episode = 0
        no_change_count = 0 # Reset no_change_count for each episode
        last_portfolio_value_check = env.initial_balance # Reset last_portfolio_value_check

        print(f"\nStarting Episode {e+1}/{episodes}...")

        while not done and step_count < max_steps_per_episode:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            if info.get('invalid_action'):
                invalid_actions += 1
            if info.get('trade_info') and info['trade_info']['action'] != 'FORCED_SELL':
                trades_this_episode += 1 # Only count agent initiated trades

            agent.remember(state, action, reward, next_state, done) # Store transition

            # Only train if enough samples are in the replay buffer
            if len(agent.memory) >= agent.batch_size:
                 agent.train()

            state = next_state
            score += reward
            step_count += 1

            # Check for minimal portfolio value change to detect stagnation
            if step_count > min_steps_before_check:
                current_portfolio_value = info.get('current_portfolio_value', env.balance + env.shares_held * info.get('current_price', env.price_data[env.current_step if env.current_step < len(env.price_data) else -1, 3]))
                if abs(current_portfolio_value - last_portfolio_value_check) < env.initial_balance * 0.0001: # Check for minimal change
                    no_change_count += 1
                else:
                    no_change_count = 0
                last_portfolio_value_check = current_portfolio_value

                if no_change_count >= no_change_threshold:
                    print(f"\n[Warning] Portfolio value has shown minimal change for {no_change_threshold} steps.")
                    print(f"Current Episode: {e+1}, Step: {step_count}")
                    print(f"Current Portfolio Value: {current_portfolio_value:.2f}")
                    print(f"Invalid Actions this episode: {invalid_actions}")
                    print(f"Trades this episode: {trades_this_episode}")
                    print(f"Current Reward: {reward:.5f}")
                    # Optionally terminate episode early for stagnation
                    # done = True
                    # print("Episode terminated early due to stagnation.")
                    # return scores, balances, invalid_action_counts, True # Indicate early stop

        scores.append(score)
        final_portfolio_value = env.balance + env.shares_held * env.price_data[env.current_step - 1, 3] if env.current_step > 0 and env.current_step <= len(env.price_data) else env.balance
        balances.append(final_portfolio_value)
        invalid_action_counts.append(invalid_actions)

        episode_time = time.time() - episode_start
        elapsed_time = time.time() - start_time
        print(f"Episode: {e+1}/{episodes}, "
              f"Score: {score:.4f}, "
              f"Final Portfolio Value: {final_portfolio_value:.2f}, "
              f"Trades: {trades_this_episode}, "
              f"Invalid Actions: {invalid_actions}, "
              f"Epsilon: {agent.epsilon:.4f}, "
              f"Beta: {agent.beta:.4f}, " # Print beta
              f"Episode Time: {episode_time:.2f}s, "
              f"Total Time: {elapsed_time:.2f}s")
        print_memory_usage()

    return scores, balances, invalid_action_counts, False

def evaluate_agent(env, agent, episodes=10):
    total_return = 0
    total_trades = 0
    total_invalid_actions = 0
    start_time = time.time()

    print("\nStarting Evaluation...")

    for e in range(episodes):
        state = env.reset()
        done = False
        episode_trades = 0
        episode_invalid_actions = 0
        episode_rewards = []

        while not done:
            # In evaluation, no exploration (epsilon=0) and no forced trades
            action = agent.act(state, training=False) # Use trained policy

            # Check for invalid action even during evaluation before stepping
            if env._is_invalid_action(action):
                 # If the optimal action from the trained policy is invalid,
                 # choose the next best valid action or a default valid action (Hold)
                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                 agent.main_network.eval()
                 with torch.no_grad():
                     q_values = agent.main_network(state_tensor)
                 agent.main_network.train()

                 # Mask invalid actions
                 masked_q_values = q_values.clone()
                 for a in range(agent.action_size):
                     if env._is_invalid_action(a):
                         masked_q_values[0, a] = -float('inf')

                 # Choose the best valid action
                 valid_action_chosen = torch.argmax(masked_q_values, dim=1).item()
                 print(f"Warning: Trained agent chose an invalid action ({action}) at step {env.current_step}. Selecting best valid action: {valid_action_chosen}")
                 action = valid_action_chosen
                 episode_invalid_actions += 1 # Count this as an invalid action attempt


            next_state, reward, done, info = env.step(action)

            # The step function now handles invalid action penalties and trade infos

            if info.get('trade_info') and info['trade_info']['action'] != 'FORCED_SELL':
                episode_trades += 1

            episode_rewards.append(reward)
            state = next_state

        # Final portfolio value calculation at the end of the episode
        final_portfolio_value = env.balance + env.shares_held * env.price_data[env.current_step - 1, 3] if env.current_step > 0 and env.current_step <= len(env.data) else env.balance
        return_rate = (final_portfolio_value - env.initial_balance) / env.initial_balance
        total_return += return_rate
        total_trades += episode_trades
        total_invalid_actions += episode_invalid_actions


        print(f"\nEvaluation Episode {e+1}/{episodes}")
        print(f"Return: {return_rate:.4f}")
        print(f"Final Balance: {env.balance:.2f}")
        print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"Agent Initiated Trades: {episode_trades}")
        print(f"Invalid Action Attempts: {episode_invalid_actions}")
        print(f"Average Reward: {np.mean(episode_rewards):.4f}")
        print(f"Total Shares Bought (Env): {env.total_buys}")
        print(f"Total Shares Sold (Env): {env.total_sales}") # Includes stop-loss/trailing sells
        print(f"Trade History Length: {len(env.trade_history)}")

    avg_return = total_return / episodes
    avg_trades = total_trades / episodes
    avg_invalid_actions = total_invalid_actions / episodes
    elapsed_time = time.time() - start_time

    print(f"\nEvaluation Summary:")
    print(f"Average Return: {avg_return:.4f}")
    print(f"Average Agent Initiated Trades per Episode: {avg_trades:.1f}")
    print(f"Average Invalid Action Attempts per Episode: {avg_invalid_actions:.1f}")
    print(f"Evaluation Time: {elapsed_time:.2f}s")

    return avg_return

def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock data
    """
    drop_cols = ['timestamp', 'target']
    file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    
    return df

def load_stock_data(ticker: str) -> pd.DataFrame:
    drop_cols = ['timestamp', 'target']
    file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cutoff = pd.Timestamp('2025-04-29 08:00:00', tz='UTC')

    df = df[df['timestamp'] >= cutoff]
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    print("프로그램 시작...")
    print_memory_usage()

    # Data loading
    data = load_stock_data('AAPL')

    # Environment creation
    env = StockTradingEnv(data, initial_balance=50000)

    # Define state and action space size
    state = env.reset()
    state_size = len(state)
    action_size = 3  # Sell (0), Hold (1), Buy (2)

    print(f"\nFinal state vector size: {state_size}")

    # Agent creation (Dueling DQN with Prioritized Replay)
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        env=env, # Pass environment for invalid action check
        learning_rate=0.0001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,  # epsilon decay rate
        epsilon_min=0.01, # Lower epsilon min for more exploitation later
        batch_size=256,
        memory_size=100000,  # Memory size for Prioritized Replay
        alpha=0.6, # Prioritized Replay alpha parameter
        beta=0.4, # Initial beta parameter
        beta_decay=0.995, # Beta decay rate
        beta_max=1.0 # Maximum beta
    )

    # Agent training
    print("\nTraining started...")
    num_episodes = 100  # Number of episodes
    scores, balances, invalid_actions, early_stop = train_agent(env, agent, episodes=num_episodes)

    # Evaluation after training, only if not stopped early
    if not early_stop:
        print("\nEvaluation started...")
        avg_return = evaluate_agent(env, agent, episodes=20)
    else:
        print("\nEvaluation skipped due to early termination during training.")

    print("\nProgram finished")
    print_memory_usage()

    # Plotting training results
    plt.figure(figsize=(15, 15))

    # Score plot
    plt.subplot(4, 1, 1)
    plt.plot(scores, label='Score')
    plt.plot(pd.Series(scores).rolling(window=10).mean(), label='Moving Average (10)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Dueling DQN with Prioritized Replay Learning Curve')
    plt.legend()

    # Balance plot
    plt.subplot(4, 1, 2)
    plt.plot(balances, label='Final Portfolio Value')
    plt.plot(pd.Series(balances).rolling(window=10).mean(), label='Moving Average (10)')
    plt.axhline(y=env.initial_balance, color='r', linestyle='--', label='Initial Balance')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Portfolio Value')
    plt.legend()

    # Invalid actions plot
    plt.subplot(4, 1, 3)
    plt.plot(invalid_actions, label='Invalid Actions Attempted')
    plt.plot(pd.Series(invalid_actions).rolling(window=10).mean(), label='Moving Average (10)')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.title('Invalid Action Attempts per Episode')
    plt.legend()

    # Epsilon and Beta plot
    epsilon_values = [agent.epsilon_min + (1.0 - agent.epsilon_min) * (agent.epsilon_decay ** i) for i in range(num_episodes)]
    # Beta values during training - need to simulate beta decay over episodes
    beta_values = [agent.beta] # Start with initial beta
    for _ in range(num_episodes - 1):
            beta_values.append(min(agent.beta_max, beta_values[-1] * agent.beta_decay))


    plt.subplot(4, 1, 4)
    plt.plot(epsilon_values, label='Epsilon')
    plt.plot(beta_values, label='Beta')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Epsilon and Beta Decay')
    plt.legend()


    plt.tight_layout()
    plt.savefig('dueling_dqn_prioritized_trading_results.png')
    plt.show()

    # Save the trained model
    agent.save("dueling_dqn_prioritized_stock_model.pth")