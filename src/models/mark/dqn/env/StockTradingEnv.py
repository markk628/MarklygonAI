import numpy as np
import pandas as pd
import random
import torch
from numpy.typing import NDArray

from src.preprocessing.data_processor import RollingWindowFeatureProcessor

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
        feature_processor: RollingWindowFeatureProcessor,
        initial_balance: int=10000, 
        transaction_fee: float=0.0015, 
        window_size: int=20,
        mode: str='train'  # 'train', 'validation', or 'test'
    ):  
        # TODO might not need reset_index
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self.data_nparray: np.ndarray = data.values
        self.close_prices_idx = data.columns.get_loc('close')
        self.feature_processor = feature_processor
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
        self.hold_penalty_base: float = -0.02         # holding penalty
        self.hold_penalty_max: float = -0.15          # maximum holding penalty
        self.profit_reward_weight: int = 50           # profit reward weight (used to scale rewards for profitable trades)
        self.loss_penalty_weight: int = 25            # loss penalty weight (used to scale penalty for loss)
        self.trade_reward: float = 0.1                # trade execution reward
        self.invalid_action_penalty: float = -1000    # penalty for taking invalid action
        self.stop_loss_penalty: float = -0.5
        self.small_hold_time_penalty: float = -0.0005 # penalty for holding 
        self.critical_loss_penalty: float = -10
        self.exceed_max_profit_threshold_penalty: float = -0.008
        self.exceed_profit_threshold_penalty: float = -0.004
        self.exceed_max_loss_threshold_penalty: float = -0.08
        self.exceed_loss_threshold_penalty: float = -0.04

        # profit/loss threshold settings
        self.profit_threshold: float = 0.01      # profit-taking threshold (profits are taken when gains exceed threshold%)
        self.loss_threshold: float = -0.01      # loss-cutting threshold (losses are cut once the position loses threshold% or more)
        self.trailing_stop: float = 0.008        # trailing stop value (trailing stop is activated after an threshold% move)
        self.max_profit_threshold: float = 0.02  # maximum profit-taking threshold
        self.max_loss_threshold: float = -0.03   # maximum loss threshold

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
        # rolling window scaling: Fit and transform on the features of the current window
        processed_features = self.feature_processor.get_state(features)
        self._feature_cache[current_idx] = processed_features
        return processed_features


    def _get_state(self) -> NDArray:
        """
        Construct the current state with normalized features and portfolio information
        """
        # features are from the window [self.current_step - self.window_size, self.current_step - 1]
        normalized_features_flattened = self._get_features(self.current_step)
        
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
            normalized_consecutive_trades,                              # constraint
            normalized_time_since_last_trade,                           # constraint
            normalized_consecutive_holds,                               # constraint
            normalized_last_portfolio_value,                            # constraint
            self.last_action if self.last_action is not None else 1,    # constraint
            normalized_trade_history_length                             # constraint
        ], dtype=np.float32)

        # Track portfolio value history
        self.price_history.append(current_price)
        self.portfolio_values.append(portfolio_value)
            
        return np.concatenate((normalized_features_flattened, portfolio_info)).astype(np.float32)
    
    
    def calculate_reward(self, action, trade_info, done):
        """
        Calculate the reward for the current step.
        Focuses on portfolio value change and penalties for undesirable actions/outcomes.
        """
        reward = 0.0
        current_price: float = self.data_nparray[self.current_step, self.close_prices_idx]
        current_portfolio_value = self.balance + self.shares_held * current_price
        # penalty for invalid actions
        if self._is_invalid_action(action):
            reward += self.invalid_action_penalty
        else:
            if action == 1:
                reward += self.small_hold_time_penalty
            portfolio_change_pct = (current_portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value if self.last_portfolio_value != 0 else self.initial_balance)
            # stop loss
            if trade_info.get('action') == 'FORCED_SELL':
                if trade_info.get('type') == 'STOP_LOSS_SELL':
                    reward += self.stop_loss_penalty * 2
                elif trade_info.get('type') == 'TRAILING_STOP_SELL':
                    reward += self.stop_loss_penalty
                elif trade_info.get('type') == 'CRITICAL_LOSS_SELL':
                    reward += self.critical_loss_penalty
            elif trade_info.get('type') == 'BUY' or trade_info.get('type') == 'SELL':
                reward += portfolio_change_pct * self.initial_balance
                if portfolio_change_pct > 0:
                    reward += self.trade_reward
                    current_profit_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0
                    for level, weight in zip(self.profit_taking_levels, self.profit_taking_weights):
                        if current_profit_pct > level:
                            reward += current_profit_pct * self.profit_reward_weight * weight
                else:
                    if portfolio_change_pct < self.max_loss_threshold:
                        reward += self.exceed_max_loss_threshold_penalty
                    elif portfolio_change_pct < self.loss_threshold:
                        reward += self.exceed_loss_threshold_penalty
            else:
                reward += self.small_hold_time_penalty
                if self.shares_held > 0 and self.position_open:
                    # apply a increasing penalty for consecutive holds to discourage inaction
                    hold_penalty = self.hold_penalty_base * (1 + 0.03 * self.consecutive_holds)
                    hold_penalty = min(hold_penalty, self.hold_penalty_max) # cap the penalty
                    reward += hold_penalty
                    if portfolio_change_pct > self.max_profit_threshold:
                        reward += self.exceed_max_profit_threshold_penalty
                    elif portfolio_change_pct > self.profit_threshold:
                        reward += self.exceed_profit_threshold_penalty
                    elif portfolio_change_pct < self.max_loss_threshold:
                        reward += self.exceed_max_loss_threshold_penalty
                    elif portfolio_change_pct < self.loss_threshold:
                        reward += self.exceed_loss_threshold_penalty
                    
        # significant bonus for ending with a profit or a large penalty for a large loss.
        if done:
            final_profit = (current_portfolio_value - self.initial_balance) / self.initial_balance
            if final_profit > 0:
                reward += final_profit * self.profit_reward_weight * 3 # higher weight for final profit
            else:
                reward += final_profit * self.loss_penalty_weight * 2 # higher penalty for final loss
                if final_profit < -0.5:
                    reward += -1

        self.last_portfolio_value = current_portfolio_value

        return np.clip(reward / 20, -1.0, 1.0) # TODO find better way to normalize

    
    def _is_invalid_action(self, action):
        """returns bool representing if the action is invalid"""
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]

        if action == 0:  # sell
            if self.shares_held <= 0:
                return True
            sell_amount = self.shares_held * current_price
            return sell_amount <= self.min_trade_amount
        elif action == 2:  # buy
            # cannot buy if no balance or already in a position (prevent averaging down and for simple trading cycle)
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
            reward += self.calculate_reward(action, trade_info, False)
        else:
            # process valid actions
            if action == 0:  # sell
                if not can_trade:
                    reward -= 0.1
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
            elif action == 2:  # buy
                if not can_trade:
                    reward = -0.1
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
                self.entry_price = current_price
                self.trailing_stop_price = current_price * (1 - self.trailing_stop)
                self.position_open = True
                self.total_shares_bought += shares_to_buy
                self.last_trade_step = self.current_step
                self.consecutive_trades += 1
                self.total_cost += cost
            reward += self.calculate_reward(action, trade_info, done)
            
        self.current_step += 1
        self.consecutive_holds = self.consecutive_holds + 1 if action == 1 else 0

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
                    trade_info = {
                        'type': 'CRITICAL_LOSS_SELL',
                        'shares': self.shares_held,
                        'price': current_price,
                        'amount': sell_amount,
                        'fee': fee,
                        'action': 'FORCED_SELL'
                    }
                else:
                    trade_info = {
                        'type': 'END_OF_EPISODE',
                        'shares': self.shares_held,
                        'price': current_price,
                        'amount': sell_amount,
                        'fee': fee,
                        'action': 'FORCED_SELL'
                    }
            
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

        self.last_action = action
        next_state = self._get_state()

        return next_state, reward, done, {
            'trade_info': trade_info,
            'invalid_action': invalid_action
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
            # num_data_features = self.data.columns.size
            # TODO make this more streamline
            num_data_features = self.feature_processor.feature_processor.n_components
            # flattened normalized features
            features_part_len = self.window_size * num_data_features
            # portfolio info count
            portfolio_part_len = 15
            self._state_size_cached = features_part_len + portfolio_part_len
        return self._state_size_cached