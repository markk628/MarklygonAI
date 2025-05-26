import numpy as np
import pandas as pd
import random
import statistics
import torch
from numpy.typing import NDArray

from src.config.config import (
    WINDOW_SIZE,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
)
from src.models.mark.dqn.utils.StateScaler import StateScaler

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
        state_scaler: StateScaler,
        initial_balance: int=INITIAL_BALANCE, 
        transaction_fee_pct: float=TRANSACTION_FEE_PERCENT, 
        window_size: int=WINDOW_SIZE,
        mode: str='train',  # 'train', 'validation', or 'test'
        use_hierarchical: bool = True
    ):  
        # TODO might be able to remove self.data
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self.steps_per_episode: int = len(data) - window_size
        self.data_nparray: np.ndarray = data.values
        self.filtered_data_nparray: np.ndarray = self.data[state_scaler.features].values
        self.state_scaler: StateScaler = state_scaler
        self.high_prices_idx = data.columns.get_loc('high')
        self.low_prices_idx = data.columns.get_loc('low')
        self.close_prices_idx = data.columns.get_loc('close')
        self.initial_balance: int = initial_balance
        self.transaction_fee_pct: float = transaction_fee_pct
        self.window_size: int = window_size
        self.mode: str = mode
        self.use_hierarchical = use_hierarchical
        self._feature_cache = {}
        self.risk_free_rate = 0.02 / 252
        
        # trading constraints
        self.max_position_size: float = 0.7                         # maximum position size (how much capitol can be used per trade)
        self.min_trade_amount: int = 300                            # minimum amount required to make a trade
        self.critical_loss_value: float = 0.5 * initial_balance
        
        # reward settings
        self.profit_reward_weight: int = 10                         # profit reward weight (used to scale rewards for profitable trades)
        self.loss_penalty_weight: int = 10                          # loss penalty weight (used to scale penalty for loss)
        self.trade_reward: float = 0.15                             # reward for trade execution
        self.successful_trade_reward: float = 0.5                   # reward for successful trade
        self.patience_reward: float = 0.05                          # reward for waiting for better opportunities
        self.efficient_capital_usage_reward: float = 0.1            # reward for efficient capital usage
        self.patience_in_position_reward: float = 0.01
        self.invalid_action_penalty: float = -1.5                   # penalty for taking invalid action
        # self.stop_loss_penalty: float = -3000                       # penalty for reaching stop loss thresholdd
        # self.critical_loss_penalty: float = -8000                   # penalty for reaching critical loss threshold
        self.small_hold_time_penalty: float = -0.0003               # penalty for holding 
        self.consecutive_hold_penalty_base: float = -0.01           # penalty for consecutive holding
        self.consecutive_hold_penalty_max: float = -0.1             # penalty for consecutive holding maximum
        self.exceed_max_profit_threshold_penalty: float = -0.012    # penalty for holding past max profit threshold
        self.exceed_profit_threshold_penalty: float = -0.006        # penalty for holding past profit threshold
        self.exceed_max_loss_threshold_penalty: float = -0.1        # penalty for holding past max loss threshold
        self.exceed_loss_threshold_penalty: float = -0.05,          # penalty for holding past loss threshold
        self.out_of_game_penalty: int = -5                          # penalty for losing all my money
        self.price_movement_alignment_bonus: float = 0.3            # bonus when action aligns with price movement
        self.quick_profit_taking_bonus: float = 0.2                 # bonus for taking profits quickly
        self.sharpe_reward_weight_positive = 5.0
        self.sharpe_penalty_weight_negative = 10.0
        self.max_sharpe_reward = 10.0
        self.max_sharpe_penalty = -20.0
        self.sharpe_no_returns_penalty = -5.0
        self.ranging_no_position_penalty_base = -0.005
        self.ranging_no_position_penalty_max = -0.05
        self.ranging_hold_penalty_delay = 10
        self.trending_up_no_position_penalty_base = -0.01
        self.trending_up_no_position_penalty_max = -0.1
        self.trending_up_hold_penalty_delay = 5
        self.losing_hold_penalty_base = -0.03

        # profit/loss threshold settings
        self.profit_threshold: float = 0.012                        # profit-taking threshold
        self.loss_threshold: float = -0.008                         # loss-cutting threshold
        self.trailing_stop_threshold: float = 0.007                 # trailing stop value
        self.max_profit_threshold: float = 0.025                    # maximum profit-taking threshold
        self.max_loss_threshold: float = -0.025                     # maximum loss threshold

        # additional profitability-related settings
        self.profit_taking_levels = [0.01, 0.015, 0.02, 0.03]       # tiered profit-taking levels
        self.profit_taking_weights = [1.0, 1.3, 1.6, 2.0]           # reward weights for each profit-taking level
        self.volatility_threshold: float = 0.018                    # volatility threshold (used to determine whether to adjust behavior during periods of high volatility)
        self.trend_following_weight: float = 1.5                    # trend-following reward weight (encourages the agent to follow the trend by increasing rewards when aligned with it)
        
        # market regime settings
        self.market_regime_weights = {
            'trending_up': 1.2,                                     # bonus for buying in uptrends
            'trending_down': 1.2,                                   # bonus for selling in downtrends
            'ranging': 0.8,                                         # reduced reward in ranging markets
            'high_volatility': 0.7                                  # reduced reward in high volatility
        }
        self.position_sizing_factor: float = 0.1                    # reward component for optimal position sizing
        
        # state variables
        self.random_starting_point: int = 0
        self.current_step: int = window_size + self.random_starting_point
        self.balance: int = initial_balance
        self.shares_held: int = 0
        self.total_trades: int = 0
        self.profitable_trades: int = 0
        self.loss_making_trades: int = 0
        self.total_shares_bought: int = 0
        self.total_shares_sold: int = 0
        self.total_cost: int = 0
        self.consecutive_trades: int = 0
        self.consecutive_holds: int = 0
        self.was_last_trade_profitable = False
        self.consecutive_profits: int = 0
        self.consecutive_losses: int = 0
        self.last_portfolio_value: int = initial_balance
        # self.last_action = None
        self.last_trade_step: int = -1
        self.entry_price: int = 0
        self.max_profit: int = 0
        self.max_loss: int = 0
        self.current_trade_duration: int = 0
        self.trailing_stop_price: int = 0
        self.current_drawdown: float = 0
        self.max_drawdown: float = 0
        self.invalid_actions: int = 0
        self.highest_price_since_buy: float = 0
        self.lowest_price_since_buy: float = 0
        self.total_profit: int = 0
        self.total_loss: int = 0
        self.successful_trade_durations = []
        self.highest_price: float = 0
        self.lowest_price: float = 0
        self.highest_portfolio_value_seen_so_far = self.initial_balance
        self.lowest_portfolio_value_seen_so_far = self.initial_balance
        self.transaction_fee: float = 0
        
        # market state
        self.market_regime: str = 'unknown'
        self.market_volatility: float = 0
        
        # portfolio metrics tracking
        self.portfolio_values = [self.initial_balance]
        self.action_history = []
        self.price_history = []
        self.reward_components = []
        
        
    def reset(self) -> NDArray:
        # start at window_size to ensure enough historical data for the first state's features
        self.current_step = self.window_size + self.random_starting_point
        if self.mode == 'train':
            self.random_starting_point = np.random.randint(0, len(self.data) * 0.75)
            self.steps_per_episode = len(self.data) - self.current_step
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.loss_making_trades = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_cost = 0
        self.consecutive_trades = 0
        self.consecutive_holds = 0
        self.was_last_trade_profitable = False
        self.consecutive_profits = 0
        self.consecutive_losses = 0
        self.last_portfolio_value = self.initial_balance
        # self.last_action = None
        self.last_trade_step = -1
        self.entry_price = 0
        self.max_profit = 0
        self.max_loss = 0
        self.current_trade_duration = 0
        self.trailing_stop_price = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.invalid_actions = 0
        self.highest_price_since_buy = 0
        self.lowest_price_since_buy = 0
        self.total_profit = 0
        self.total_loss = 0
        self.successful_trade_durations = []
        self.highest_price = 0
        self.lowest_price = 0
        self.highest_portfolio_value_seen_so_far = self.initial_balance
        self.lowest_portfolio_value_seen_so_far = self.initial_balance
        self.transaction_fee = 0
        self.market_regime = 'unknown'
        self.market_volatility = 0
        self.portfolio_values = [self.initial_balance]
        self.price_history = []
        self.reward_components = []
        
        return self._get_state()
    
    
    def _detect_market_regime(self) -> tuple[str, float]:
        """
        Detect current market regime (trending up, trending down, ranging) and volatility
        """
        if self.current_step < self.window_size + 10:
            return 'unknown', 0.0
        
        # get recent close prices
        recent_window = 20  # use last 20 minutes to detect regime
        start_idx = max(0, self.current_step - recent_window)
        end_idx = self.current_step
        close_prices = self.data_nparray[start_idx:end_idx, self.close_prices_idx]
        
        short_ma = np.mean(close_prices[-5:])
        long_ma = np.mean(close_prices)
        volatility = np.std(close_prices) / np.mean(close_prices)
        
        # determine market regime
        if short_ma > long_ma * 1.005:
            regime = 'trending_up'
        elif short_ma < long_ma * 0.995:
            regime = 'trending_down'
        else:
            regime = 'ranging'
            
        # if volatility is high, override with high_volatility regime
        if volatility > self.volatility_threshold:
            regime = 'high_volatility'
            
        return regime, volatility
    
    
    def _get_features(self, current_idx: int) -> NDArray:
        """
        Extract features from a rolling window of historical data
        """
        # return cached data for current_idx if it exists
        if current_idx in self._feature_cache:
            return self._feature_cache[current_idx]
                
        start_idx: int = current_idx - self.window_size
        end_idx: int = current_idx
        window = self.filtered_data_nparray[start_idx:end_idx]
        # rolling window scaling: fit and transform on the features of the current window
        processed_window = self.state_scaler.scale_stock_data(window).flatten()
        self._feature_cache[current_idx] = processed_window
        return processed_window
    
    
    def _calculate_price_acceleration(self) -> float:
        """
        Calculate price acceleration (2nd derivative of price)
        This measures how the rate of price change is changing
        """
        if self.current_step < 2:
            return 0.0
        
        # get the last 3 close prices
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        prev_price = self.data_nparray[self.current_step - 1, self.close_prices_idx]
        prev_prev_price = self.data_nparray[self.current_step - 2, self.close_prices_idx]
        
        # calculate first velocities
        velocity_current = (current_price - prev_price) / prev_price
        velocity_previous = (prev_price - prev_prev_price) / prev_prev_price
        
        # calculate acceleration then normalize
        acceleration = velocity_current - velocity_previous
        return np.clip(acceleration * 1000, -1, 1)
    
    
    def _calculate_volatility_trend(self, window=WINDOW_SIZE) -> float:
        """
        Calculate volatility trend (whether volatility is increasing or decreasing)
        Returns: -1 (decreasing volatility) to 1 (increasing volatility)
        """
        if self.current_step < window * 2:
            return 0.0
        
        # calculate recent volatility (last 'window' periods)
        recent_start = max(0, self.current_step - window + 1)
        recent_prices = self.data_nparray[recent_start:self.current_step + 1, self.close_prices_idx]
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        recent_volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
        
        # calculate older volatility (previous 'window' periods)
        older_start = max(0, self.current_step - window * 2 + 1)
        older_end = self.current_step - window + 1
        if older_end > older_start:
            older_prices = self.data_nparray[older_start:older_end, self.close_prices_idx]
            older_returns = np.diff(older_prices) / older_prices[:-1]
            older_volatility = np.std(older_returns) if len(older_returns) > 1 else 0
        else:
            older_volatility = recent_volatility
        
        # calculate volatility trend
        if older_volatility > 0:
            volatility_change = (recent_volatility - older_volatility) / older_volatility
            # normalize to -1 to 1 range
            return np.clip(volatility_change * 5, -1, 1)  # multiply by 5 for sensitivity
        return 0.0
    
    
    def _calculate_price_relative_to_range_hilo(self, lookback_window=WINDOW_SIZE) -> float:
        """
        Alternative implementation using high/low data for more accurate range calculation
        Assumes self.high_prices_idx and self.low_prices_idx are available
        """
        if self.current_step < lookback_window:
            start_idx = 0
        else:
            start_idx = self.current_step - lookback_window + 1
        
        # get high and low values for the lookback window
        high_window = self.data_nparray[start_idx:self.current_step + 1, self.high_prices_idx]
        low_window = self.data_nparray[start_idx:self.current_step + 1, self.low_prices_idx]
        
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        range_high = np.max(high_window)
        range_low = np.min(low_window)
        
        if range_high == range_low:
            return 0.5
        
        relative_position = (current_price - range_low) / (range_high - range_low)
        return np.clip(relative_position, 0, 1)
    
    
    def _is_out_of_game(self):
        return self.shares_held == 0 and self.balance < self.initial_balance * 0.5
    
    
    def _get_state(self) -> NDArray:
        # normalized features
        normalized_features_flattened = self._get_features(self.current_step)
        
        # portfolio metrics
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        portfolio_value = self.balance + self.shares_held * current_price
        balance_ratio = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        shares_value = self.shares_held * current_price
        shares_value_ratio = (self.shares_held * current_price) / self.initial_balance if self.initial_balance > 0 else 0
        is_out_of_game = self._is_out_of_game()
        
        # performance metrics
        avg_buy_price = self.total_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0
        position_pl = (current_price - avg_buy_price) * self.shares_held if self.shares_held > 0 else 0
        position_pl_ratio = position_pl / (self.total_cost if self.total_cost > 0 else 1)
        position_pl_ratio_initial_balance = position_pl / self.initial_balance if self.initial_balance > 0 else 0
        win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0
        avg_win = self.total_profit / self.profitable_trades if self.profitable_trades > 0 else 0
        avg_loss = self.total_loss / self.loss_making_trades if self.loss_making_trades > 0 else 1e-6
        win_loss_ratio = self.profitable_trades / self.loss_making_trades if self.loss_making_trades > 0 else 1.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        if len(self.portfolio_values) > 20:
            recent_returns = [(self.portfolio_values[i] / self.portfolio_values[i-1]) - 1 
                            for i in range(max(0, len(self.portfolio_values)-20), len(self.portfolio_values))
                            if i > 0]
            if recent_returns:
                avg_return = statistics.mean(recent_returns)
                std_return = statistics.stdev(recent_returns) if statistics.stdev(recent_returns) > 0 else 1e-6
                sharpe_ratio = avg_return / std_return  # Simplified Sharpe (no risk-free rate)
                
                # sortino ratio (only considering negative returns/downside deviation)
                negative_returns = [r for r in recent_returns if r < 0]
                downside_std = np.std(negative_returns) if negative_returns and np.std(negative_returns) > 0 else 1e-6
                sortino_ratio = avg_return / downside_std if downside_std > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # risk metrics
        peak_value = max(self.portfolio_values)
        self.current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        proximity_to_critical_loss = portfolio_value - self.critical_loss_value
        highest_price_since_buy_and_entry_price_ratio = self.highest_price_since_buy / self.entry_price if self.entry_price > 0 else 0
        lowest_price_since_buy_and_entry_price_ratio = self.lowest_price_since_buy / self.entry_price if self.entry_price > 0 else 0
        potential_profit_ratio = (highest_price_since_buy_and_entry_price_ratio - 1) if self.shares_held > 0 else 0
        potential_loss_ratio = (1 - lowest_price_since_buy_and_entry_price_ratio) if self.shares_held > 0 else 0
        
        # market state metrics
        self.market_regime, self.market_volatility = self._detect_market_regime()
        trend_direction = 0
        if self.market_regime == 'trending_up':
            trend_direction = 1
        elif self.market_regime == 'trending_down':
            trend_direction = -1
        recent_price_change = (current_price / self.data_nparray[self.current_step-5, self.close_prices_idx]) - 1 if self.current_step > 5 else 0
        highest_price_and_current_price_ratio = self.highest_price / current_price if current_price > 0 else 0
        highest_price_and_entry_price_ratio = self.highest_price / self.entry_price if self.entry_price > 0 else 0
        lowest_price_and_current_price_ratio = self.lowest_price / current_price if current_price > 0 else 0
        lowest_price_and_entry_price_ratio = self.lowest_price / self.entry_price if self.entry_price > 0 else 0
        
        # position management metrics
        # kelly_fraction = max(0, min(1, (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio)) if win_loss_ratio > 0 else 0
        # optimal_position_size = kelly_fraction * self.max_position_size
        # position_utilization_kelly = (self.shares_held * current_price) / (self.initial_balance * optimal_position_size) if self.initial_balance > 0 and optimal_position_size > 0 else 0
        # position_utilization_max = (self.shares_held * current_price) / (self.initial_balance * self.max_position_size) if self.initial_balance > 0 else 0
        highest_price_since_buy_and_current_price_ratio = self.highest_price_since_buy / current_price if current_price > 0 else 0
        lowest_price_since_buy_and_current_price_ratio = self.lowest_price_since_buy / current_price if current_price > 0 else 0
        is_buy_invalid_next_step = 1 if self.shares_held > 0 or is_out_of_game else 0
        is_sell_invalid_next_step = 1 if self.shares_held <= 0 else 0
        
        # trading behavior metricss
        time_in_position = self.current_trade_duration / self.steps_per_episode if self.shares_held > 0 else 0
        optimal_holding_time = statistics.mean(self.successful_trade_durations) / self.steps_per_episode if len(self.successful_trade_durations) > 0 else 0.1
        time_ratio_to_optimal = time_in_position / optimal_holding_time if optimal_holding_time > 0 else 0
        time_since_last_trade = np.clip((self.current_step - self.last_trade_step) / self.steps_per_episode, 0, 1)
        # last_action = float(self.last_action) / 2 if self.last_action is not None else 0.5
        

        # track portfolio value and price history
        self.price_history.append(current_price)
        self.portfolio_values.append(portfolio_value)
        
        ###########################################################################################
        ######################################### READ ME #########################################
        ###########################################################################################
        # if any of the metrics below changes update the metric's value in self.get_branch_sizes()
        
        if self.use_hierarchical:
            raw_states = {
                # portfolio metrics
                'portfolio_value': portfolio_value,
                'last_portfolio_value': self.last_portfolio_value,
                'balance': self.balance,
                'balance_ratio': balance_ratio,
                'shares': self.shares_held,
                'shares_value': shares_value,
                'shares_value_ratio': shares_value_ratio,
                'is_out_of_game': float(is_out_of_game),
                
                # performance metrics
                'average_buy_price': avg_buy_price,
                'position_pl': position_pl,
                'position_pl_total_cost_ratio': position_pl_ratio,
                'position_pl_ratio_initial_balance': position_pl_ratio_initial_balance,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'avg_win_loss_ratio': avg_win_loss_ratio,
                'consecutive_profits': self.consecutive_profits,
                'consecutive_losses': self.consecutive_losses,
                'sharpe': sharpe_ratio,
                'sortino': sortino_ratio,
                'was_last_trade_profitable': float(self.was_last_trade_profitable),
                'transaction_fee': self.transaction_fee,

                # risk metrics
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'proximity_to_critical_loss': proximity_to_critical_loss,
                'potential_profit_ratio': potential_profit_ratio,
                'potential_loss_ratio': potential_loss_ratio,
                
                # market state metrics
                'is_trending_up': float(self.market_regime == 'trending_up'),    
                'is_trending_down': float(self.market_regime == 'trending_down'),  
                'is_ranging': float(self.market_regime == 'ranging'),        
                'is_volatile': float(self.market_regime == 'high_volatility'),
                'unknown': float(self.market_regime == 'unknown'),
                'trend_direction': trend_direction,
                'volatility': self.market_volatility,
                'recent_price_change': recent_price_change,
                'price_acceleration': self._calculate_price_acceleration(),
                'volatility_trend': self._calculate_volatility_trend(),
                'price_relative_to_range_hilo': self._calculate_price_relative_to_range_hilo(),
                'highest_price_and_current_price_ratio': highest_price_and_current_price_ratio,
                'highest_price_and_entry_price_ratio': highest_price_and_entry_price_ratio,
                'lowest_price_and_current_price_ratio': lowest_price_and_current_price_ratio,
                'lowest_price_and_entry_price_ratio': lowest_price_and_entry_price_ratio,
                
                # position management metrics
                # 'position_utilization_kelly': position_utilization_kelly,
                # 'position_utilization_max': position_utilization_max,
                'invalid_actions': self.invalid_actions,
                'highest_price_since_buy_and_entry_price_ratio': highest_price_since_buy_and_entry_price_ratio,
                'highest_price_since_buy_and_current_price_ratio': highest_price_since_buy_and_current_price_ratio,
                'lowest_price_since_buy_and_entry_price_ratio': lowest_price_since_buy_and_entry_price_ratio,
                'lowest_price_since_buy_and_current_price_ratio': lowest_price_since_buy_and_current_price_ratio,
                'is_buy_invalid_next_step': is_buy_invalid_next_step,
                'is_sell_invalid_next_step': is_sell_invalid_next_step,
                
                # trading behavior metrics
                'time_in_position': time_in_position,
                'optimal_holding_time': optimal_holding_time,
                'time_ratio_to_optimal': time_ratio_to_optimal,
                'time_since_last_trade': time_since_last_trade,
                'consecutive_holds': self.consecutive_holds,
                'consecutive_trades': self.consecutive_trades,
                'total_trades': self.total_trades,
                # 'last_action': last_action
            }
            
            normalized_states = self.state_scaler.scale_state_vector(raw_states)
            
            temporal_metrics = np.array([
                self.data_nparray[self.current_step, self.data.columns.get_loc('minute_sin')],
                self.data_nparray[self.current_step, self.data.columns.get_loc('minute_cos')],
                self.data_nparray[self.current_step, self.data.columns.get_loc('hour_sin')],
                self.data_nparray[self.current_step, self.data.columns.get_loc('hour_cos')],
                self.data_nparray[self.current_step, self.data.columns.get_loc('day_sin')],
                self.data_nparray[self.current_step, self.data.columns.get_loc('day_cos')],
                # TODO use when running 2 years worth of data
                # self.data_nparray[self.current_step, self.data.columns.get_loc('month_sin')],
                # self.data_nparray[self.current_step, self.data.columns.get_loc('month_cos')],
                # self.data_nparray[self.current_step, self.data.columns.get_loc('quarter_sin')],
                # self.data_nparray[self.current_step, self.data.columns.get_loc('quarter_cos')]
            ], dtype=np.float32)
            
            return np.concatenate((
                normalized_features_flattened, 
                normalized_states,
                temporal_metrics,
            )).astype(np.float32)
            
        raw_states = {
            # portfolio metrics
            'portfolio_value': portfolio_value,
            'last_portfolio_value': self.last_portfolio_value,
            'balance': self.balance,
            'balance_ratio': balance_ratio,
            'shares': self.shares_held,
            'shares_value': shares_value,
            'shares_value_ratio': shares_value_ratio,
            'average_buy_price': avg_buy_price,
            'position_pl': position_pl,
            'position_pl_total_cost_ratio': position_pl_ratio,
            'position_pl_ratio_initial_balance': position_pl_ratio_initial_balance,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'consecutive_profits': self.consecutive_profits,
            'consecutive_losses': self.consecutive_losses,
            'sharpe': sharpe_ratio,
            'sortino': sortino_ratio,
            'was_last_trade_profitable': float(self.was_last_trade_profitable),
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'proximity_to_critical_loss': proximity_to_critical_loss,
            'potential_profit_ratio': potential_profit_ratio,
            'potential_loss_ratio': potential_loss_ratio,
            # 'position_utilization_kelly': position_utilization_kelly,
            # 'position_utilization_max': position_utilization_max,
            'invalid_actions': self.invalid_actions,
            'highest_price_since_buy_and_entry_price_ratio': highest_price_since_buy_and_entry_price_ratio,
            'highest_price_since_buy_and_current_price_ratio': highest_price_since_buy_and_current_price_ratio,
            'lowest_price_since_buy_and_entry_price_ratio': lowest_price_since_buy_and_entry_price_ratio,
            'lowest_price_since_buy_and_current_price_ratio': lowest_price_since_buy_and_current_price_ratio,
            'transaction_fee': self.transaction_fee,
            
            # market state metrics
            'is_trending_up': float(self.market_regime == 'trending_up'),    
            'is_trending_down': float(self.market_regime == 'trending_down'),  
            'is_ranging': float(self.market_regime == 'ranging'),        
            'is_volatile': float(self.market_regime == 'high_volatility'),
            'unknonw': float(self.market_regime == 'unknown'),
            'trend_direction': trend_direction,
            'volatility': self.market_volatility,
            'recent_price_change': recent_price_change,
            'price_acceleration': self._calculate_price_acceleration(),
            'volatility_trend': self._calculate_volatility_trend(),
            'price_relative_to_range_hilo': self._calculate_price_relative_to_range_hilo(),
            'highest_price_and_current_price_ratio': highest_price_and_current_price_ratio,
            'highest_price_and_entry_price_ratio': highest_price_and_entry_price_ratio,
            'lowest_price_and_current_price_ratio': lowest_price_and_current_price_ratio,
            'lowest_price_and_entry_price_ratio': lowest_price_and_entry_price_ratio,
            
            # constraints
            'is_out_of_game': float(is_out_of_game),
            'is_buy_invalid_next_step': is_buy_invalid_next_step,
            'is_sell_invalid_next_step': is_sell_invalid_next_step,
            'time_in_position': time_in_position,
            'optimal_holding_time': optimal_holding_time,
            'time_ratio_to_optimal': time_ratio_to_optimal,
            'time_since_last_trade': time_since_last_trade,
            'consecutive_holds': self.consecutive_holds,
            'consecutive_trades': self.consecutive_trades,
            'total_trades': self.total_trades,
            # 'last_action': last_action
        }
        
        normalized_states = self.state_scaler.scale_state_vector(raw_states)

        return np.concatenate((
            normalized_features_flattened, 
            normalized_states
        )).astype(np.float32)
    
    
    def _calculate_reward(self, is_invalid, action, trade_info, done):
        """
        Calculate the reward for the current step
        """
        reward_components = {}
        total_reward = 0.0
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        portfolio_value = self.balance + self.shares_held * current_price
        
        # end of episode state additional rewards/penalties
        if done:
            final_profit = (portfolio_value - self.initial_balance) / self.initial_balance
            
            # reward/penalty based on final performance
            if final_profit > 0:
                # stronger reward for ending with profit
                final_reward = final_profit * self.profit_reward_weight * 3
                reward_components['final_profit'] = final_reward
                total_reward += final_reward
                
                # bonus based on consistency
                if self.profitable_trades > self.loss_making_trades * 1.5:  # at least 60% win rate
                    consistency_reward = final_profit * 0.5
                    reward_components['consistency'] = consistency_reward
                    total_reward += consistency_reward
            else:
                # penalty for ending with a loss
                final_penalty = final_profit * self.loss_penalty_weight * 2
                reward_components['final_loss'] = final_penalty
                total_reward += final_penalty
                
                # extra penalty for severe losses
                if final_profit < -0.3:
                    severe_loss_penalty = -5 * abs(final_profit)
                    reward_components['severe_loss'] = severe_loss_penalty
                    total_reward += severe_loss_penalty
                    
            # sharpe ratio reward/penalty
            sharpe_ratio = self._calculate_sharpe_ratio()
            sharpe_reward_component = 0.0
            if sharpe_ratio >= 0:
                sharpe_reward_component = sharpe_ratio * self.sharpe_reward_weight_positive
            else: # negative Sharpe Ratio
                sharpe_reward_component = sharpe_ratio * self.sharpe_penalty_weight_negative
            sharpe_reward_component = np.clip(sharpe_reward_component, self.max_sharpe_penalty, self.max_sharpe_reward)
            reward_components['sharpe_ratio_reward'] = sharpe_reward_component
            total_reward += sharpe_reward_component
                
            # capital efficiency reward
            trade_frequency = self.total_trades / (self.steps_per_episode / 2)
            if trade_frequency > 0.1 and final_profit > 0:  # reward active trading if profitable
                capital_efficiency = self.efficient_capital_usage_reward * trade_frequency * 10
                reward_components['capital_efficiency'] = capital_efficiency
                total_reward += capital_efficiency
            
            if self.invalid_actions > 0:
                reward_components['invalid_actions_scaling_factor'] = self.invalid_actions * self.invalid_action_penalty
                total_reward += reward_components['invalid_actions_scaling_factor'] 
            
            if self._is_out_of_game():
                reward_components['out_of_game'] = self.out_of_game_penalty
                total_reward += reward_components['out_of_game']
        else:
            # invalid action penalty
            if is_invalid:
                reward_components['invalid_action'] = self.invalid_action_penalty
                total_reward += reward_components['invalid_action']
            else:
                # base portfolio change component           
                portfolio_change_pct = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
                reward_components['portfolio_change'] = portfolio_change_pct * 2
                total_reward += reward_components['portfolio_change']
                
                # overall profitability component
                overall_profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance
                reward_components['overall_profit'] = overall_profit_pct
                total_reward += reward_components['overall_profit']
                
                # current drawdown penalty
                if self.highest_portfolio_value_seen_so_far > 0:
                    current_drawdown = 1 - portfolio_value / self.highest_portfolio_value_seen_so_far
                else:
                    current_drawdown = 0
                reward_components['step_drawdown_penalty'] = current_drawdown
                total_reward += current_drawdown
                if trade_info.get('type') == 'BUY':
                    # reward/penalty based on market regime alignment
                    if self.market_regime == 'trending_up':
                        # bonus for buying in uptrend
                        reward_components['trend_alignment'] = self.trade_reward * self.market_regime_weights['trending_up']
                        total_reward += reward_components['trend_alignment']
                    elif self.market_regime == 'trending_down':
                        # penalty for buying in downtrend
                        reward_components['trend_alignment'] = -self.trade_reward * 0.5
                        total_reward += reward_components['trend_alignment']
                    
                    # base trade reward
                    reward_components['trade_execution'] = self.trade_reward
                    total_reward += reward_components['trade_execution']
                    
                    # position sizing component
                    optimal_position_size = self.initial_balance * self.max_position_size
                    actual_position_size = self.shares_held * current_price
                    position_sizing_ratio = min(actual_position_size / optimal_position_size, 1.0) if optimal_position_size > 0 else 0
                    reward_components['position_sizing'] = position_sizing_ratio * self.position_sizing_factor
                    total_reward += reward_components['position_sizing']
                    
                elif trade_info.get('type') == 'SELL':
                    # calculate profit/loss from this trade
                    buy_fee = self.entry_price * self.shares_held * self.transaction_fee_pct
                    profit_pct = (trade_info['sell_amount'] / trade_info['buy_amount']) - buy_fee # if trade_info['entry_price'] > 0 and trade_info['shares'] > 0 else 0
                    
                    # reward based on profit
                    if profit_pct > 0:
                        # scale reward based on profit percentage
                        profit_reward = profit_pct * self.profit_reward_weight
                        
                        # additional reward for quick profitable trades
                        if trade_info['trade_duration'] < self.steps_per_episode * 0.3:  # less than 30% of possible trade steps
                            profit_reward *= 1.2  # 20% bonus for quick profits
                            reward_components['quick_profit'] = self.quick_profit_taking_bonus
                            total_reward += reward_components['quick_profit']
                            
                        reward_components['profit_reward'] = profit_reward
                        total_reward += reward_components['profit_reward']
                        
                        # additional tiered reward based on profit levels
                        for level, weight in zip(self.profit_taking_levels, self.profit_taking_weights):
                            if profit_pct > level:
                                tier_reward = profit_pct * weight
                                reward_components[f'profit_tier_{level}'] = tier_reward
                                total_reward += tier_reward
                    else:
                        # penalty based on loss
                        loss_penalty = profit_pct * self.loss_penalty_weight
                        reward_components['loss_penalty'] = loss_penalty
                        total_reward += loss_penalty
                    
                    # trend alignment for selling
                    if self.market_regime == 'trending_down':
                        # Bonus for selling in downtrend
                        reward_components['trend_alignment'] = self.trade_reward * self.market_regime_weights['trending_down']
                        total_reward += reward_components['trend_alignment']
                        
                    # base trade reward
                    reward_components['trade_execution'] = self.trade_reward
                    total_reward += reward_components['trade_execution']
                    
                else:  # HOLD action
                    # small continuous penalty for holding to encourage decisive action
                    reward_components['small_hold_time_penalty'] = self.small_hold_time_penalty
                    total_reward += reward_components['small_hold_time_penalty']
                    
                    if self.shares_held > 0:
                        current_position_pct = (current_price - self.entry_price) / self.entry_price
                        
                        # holding a profitable position
                        if current_position_pct >= self.profit_threshold:
                            patience_in_position_reward = self.patience_in_position_reward * (1 + 0.05 * self.consecutive_holds)
                            # bonus if market is trending up
                            if self.market_regime == 'trending_up':
                                patience_in_position_reward *= 1.2
                            elif self.market_regime == 'ranging':
                                patience_in_position_reward *= 1.1

                            reward_components['patience_in_position_profit'] = patience_in_position_reward
                            total_reward += patience_in_position_reward

                        elif current_position_pct > self.max_loss_threshold: # not at max loss threshold yet
                            patience_in_position_reward = self.patience_in_position_reward * 0.5 * (1 + 0.01 * self.consecutive_holds)
                            if self.market_regime == 'ranging':
                                patience_in_position_reward *= 1.1
                            elif self.market_regime == 'trending_up':
                                patience_in_position_reward *= 0.8
                            reward_components['patience_in_position_neutral'] = patience_in_position_reward
                            total_reward += patience_in_position_reward
                        
                        # penalty for holding a losing position, especially if the market is trending against it.
                        # hopefully discourages holding onto losses.
                        if current_position_pct < 0:
                            losing_hold_penalty = self.losing_hold_penalty_base * (1 + 0.03 * self.consecutive_holds)
                            if self.market_regime == 'trending_down':
                                losing_hold_penalty *= 1.5
                            reward_components['losing_position_hold_penalty'] = losing_hold_penalty
                            total_reward += reward_components['losing_position_hold_penalty']
                        
                        # penalties for holding beyond thresholds
                        if current_position_pct > self.max_profit_threshold:
                            reward_components['exceed_max_profit'] = self.exceed_max_profit_threshold_penalty
                            total_reward += reward_components['exceed_max_profit']
                        elif current_position_pct > self.profit_threshold:
                            reward_components['exceed_profit'] = self.exceed_profit_threshold_penalty
                            total_reward += reward_components['exceed_profit']
                        elif current_position_pct < self.max_loss_threshold:
                            reward_components['exceed_max_loss'] = self.exceed_max_loss_threshold_penalty
                            total_reward += reward_components['exceed_max_loss']
                        elif current_position_pct < self.loss_threshold:
                            reward_components['exceed_loss'] = self.exceed_loss_threshold_penalty
                            total_reward += reward_components['exceed_loss']
                    else:
                        # patience reward for waiting while having no position (only if we have enough balance)
                        if self.balance > self.initial_balance * 0.5 and self.consecutive_holds > 5:
                            patience_reward = min(self.patience_reward * (self.consecutive_holds / 20), self.patience_reward * 2)
                            if self.market_regime == 'trending_down':
                                patience_reward *= 1.3
                            reward_components['patience'] = patience_reward
                            total_reward += patience_reward  
                        if self.balance > self.initial_balance * 0.8: # only if there's still a lot of money
                            if self.market_regime == 'ranging' and self.consecutive_holds > self.ranging_hold_penalty_delay:
                                ranging_no_position_penalty = self.ranging_no_position_penalty_base * (1 + 0.01 * self.consecutive_holds)
                                ranging_no_position_penalty = max(ranging_no_position_penalty, self.ranging_no_position_penalty_max)
                                reward_components['ranging_no_position_penalty'] = ranging_no_position_penalty
                                total_reward += reward_components['ranging_no_position_penalty']

                            elif self.market_regime == 'trending_up' and self.consecutive_holds > self.trending_up_hold_penalty_delay:
                                trending_up_no_position_penalty = self.trending_up_no_position_penalty_base * (1 + 0.02 * self.consecutive_holds)
                                trending_up_no_position_penalty = max(trending_up_no_position_penalty, self.trending_up_no_position_penalty_max)
                                reward_components['trending_up_no_position_penalty'] = trending_up_no_position_penalty
                                total_reward += reward_components['trending_up_no_position_penalty']

        self.reward_components.append(reward_components)
        return total_reward[0] if isinstance(total_reward, np.ndarray) else total_reward

    # def _calculate_reward(self, is_invalid, action, trade_info, done):
    #     """
    #     Calculate the reward for the current step with improved stability and clarity
    #     """
    #     reward_components = {'action': action}
    #     total_reward = 0.0
    #     current_price = self.data_nparray[self.current_step, self.close_prices_idx]
    #     portfolio_value = self.balance + self.shares_held * current_price
        
    #     # Ensure portfolio_value is always a scalar
    #     if isinstance(portfolio_value, np.ndarray):
    #         portfolio_value = float(portfolio_value.item())
        
    #     # Base metrics
    #     portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
    #     portfolio_change = (portfolio_value - self.last_portfolio_value) / max(self.last_portfolio_value, 1e-8)
    #     if is_invalid and not done:
    #         reward_components['invalid_action'] = self.invalid_action_penalty
    #         total_reward += self.invalid_action_penalty
    #     elif done:
    #         total_reward += self._calculate_terminal_rewards(portfolio_return, reward_components)
    #     else:
    #         total_reward += self._calculate_step_rewards(action, 
    #                                                      trade_info, 
    #                                                      portfolio_value,
    #                                                      portfolio_change, 
    #                                                      portfolio_return,
    #                                                      current_price,
    #                                                      reward_components)
        
    #     # Ensure total_reward is scalar and bounded
    #     total_reward = float(np.clip(total_reward, -10.0, 10.0))
    #     reward_components['trade_info'] = trade_info
    #     reward_components['balance'] = self.balance
    #     reward_components['shares'] = self.shares_held
    #     reward_components['portfolio_value'] = portfolio_value
    #     reward_components['reward'] = total_reward
    #     self.reward_components.append(reward_components)
    #     return total_reward

    # def _calculate_terminal_rewards(self, portfolio_return, reward_components):
    #     """Calculate rewards at episode termination"""
    #     terminal_reward = 0.0
        
    #     # Final performance reward (normalized)
    #     final_return_reward = np.tanh(portfolio_return * 5)  # Bounded between -1 and 1
    #     reward_components['final_return'] = final_return_reward
    #     terminal_reward += final_return_reward * 2.0
        
    #     # Sharpe ratio reward (bounded)
    #     sharpe_ratio = self._calculate_sharpe_ratio()
    #     if not np.isnan(sharpe_ratio) and not np.isinf(sharpe_ratio): 
    #         sharpe_reward = np.tanh(sharpe_ratio) * 1.0
    #         reward_components['sharpe_ratio'] = sharpe_reward
    #         terminal_reward += sharpe_reward
        
    #     # Trading consistency reward
    #     if self.total_trades > 0:
    #         win_rate = self.profitable_trades / max(self.total_trades, 1)
    #         consistency_reward = (win_rate - 0.5) * 1.0  # Reward above 50% win rate
    #         reward_components['consistency'] = consistency_reward
    #         terminal_reward += consistency_reward
        
    #     # Penalty for invalid actions
    #     if self.invalid_actions > 0:
    #         invalid_penalty = -min(self.invalid_actions * 0.1, 2.0)  # Capped penalty
    #         reward_components['invalid_actions'] = invalid_penalty
    #         terminal_reward += invalid_penalty
        
    #     # Out of game penalty
    #     if self._is_out_of_game():
    #         reward_components['out_of_game'] = -5.0
    #         terminal_reward -= 5.0
        
    #     return terminal_reward

    # def _calculate_step_rewards(self,
    #                             action, 
    #                             trade_info,
    #                             portfolio_value,
    #                             portfolio_change, 
    #                             portfolio_return,
    #                             current_price,
    #                             reward_components):
    #     """Calculate rewards for individual steps"""
    #     step_reward = 0.0
        
    #     # Portfolio change reward (immediate feedback)
    #     # portfolio_reward = np.tanh(portfolio_change * 20) * 0.5  # Bounded and scaled
    #     # reward_components['portfolio_change'] = portfolio_reward
    #     # step_reward += portfolio_reward
        
    #     # # Overall return signal (weaker but consistent)
    #     # return_signal = np.tanh(portfolio_return * 10) * 0.2
    #     # reward_components['return_signal'] = return_signal
    #     # step_reward += return_signal
        
    #     # Action-specific rewards
    #     if trade_info.get('type') == 'BUY':
    #         step_reward += self._calculate_buy_rewards(current_price, reward_components)
    #     elif trade_info.get('type') == 'SELL':
    #         step_reward += self._calculate_sell_rewards(trade_info, reward_components)
    #     else:  # HOLD
    #         step_reward += self._calculate_hold_rewards(current_price, reward_components)
        
    #     # Drawdown penalty (bounded)
    #     # if self.highest_portfolio_value_seen_so_far > 0:
    #     #     drawdown = (self.highest_portfolio_value_seen_so_far - portfolio_value) / self.highest_portfolio_value_seen_so_far
    #     #     drawdown_penalty = -min(drawdown * 2, 1.0)  # Max penalty of -1
    #     #     reward_components['drawdown'] = drawdown_penalty
    #     #     step_reward += drawdown_penalty
        
    #     return step_reward

    # def _calculate_buy_rewards(self, current_price, reward_components):
    #     """Calculate rewards for buy actions"""
    #     buy_reward = 0.0
        
    #     # Base trade execution reward
    #     buy_reward += 0.001
    #     reward_components['buy_execution'] = 0.001
        
    #     # Market regime alignment
    #     if hasattr(self, 'market_regime'):
    #         if self.market_regime == 'trending_up':
    #             regime_bonus = 0.1
    #             reward_components['trend_alignment'] = regime_bonus
    #             buy_reward += regime_bonus
    #         elif self.market_regime == 'trending_down':
    #             regime_penalty = -0.05
    #             reward_components['trend_misalignment'] = regime_penalty
    #             buy_reward += regime_penalty
        
    #     return buy_reward

    # def _calculate_sell_rewards(self, trade_info, reward_components):
    #     """Calculate rewards for sell actions"""
    #     sell_reward = 0.0
        
    #     # Base trade execution reward
    #     sell_reward += 0.001
    #     reward_components['sell_execution'] = 0.001
        
    #     entry_cost = trade_info['buy_amount']
    #     exit_amount = trade_info['sell_amount']
        
    #     total_return = (exit_amount - entry_cost) / (entry_cost)
        
    #     # Bounded profit/loss reward
    #     profit_reward = np.tanh(total_return * 10) * 1.0
    #     reward_components['trade_profit'] = profit_reward
    #     sell_reward += profit_reward
        
    #     # Quick profit bonus
    #     if (total_return > 0 and 
    #         trade_info.get('current_trade_duration', float('inf')) < self.steps_per_episode * 0.3):
    #         quick_bonus = 0.2
    #         reward_components['quick_profit'] = quick_bonus
    #         sell_reward += quick_bonus
        
    #     return sell_reward

    # def _calculate_hold_rewards(self, current_price, reward_components):
    #     """Calculate rewards for hold actions"""
    #     hold_reward = 0.0
        
    #     # Small base penalty to encourage action
    #     hold_penalty = -0.01
    #     reward_components['hold_penalty'] = hold_penalty
    #     hold_reward += hold_penalty
        
    #     if self.shares_held > 0:
    #         # Position-based rewards
    #         position_return = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
            
    #         # Reward holding profitable positions (with diminishing returns)
    #         if position_return > 0.02:  # 2% profit threshold
    #             patience_reward = min(0.1 * np.log(1 + position_return), 0.3)
    #             reward_components['profitable_hold'] = patience_reward
    #             hold_reward += patience_reward
            
    #         # Penalty for holding large losses
    #         elif position_return < -0.05:  # 5% loss threshold
    #             loss_penalty = max(-0.2 * abs(position_return), -0.5)
    #             reward_components['loss_hold'] = loss_penalty
    #             hold_reward += loss_penalty
        
    #     else:
    #         # No position - patience reward in bad markets
    #         if (hasattr(self, 'market_regime') and 
    #             self.market_regime == 'trending_down' and 
    #             self.consecutive_holds > 10):
    #             patience_reward = min(0.05 * (self.consecutive_holds - 10) / 20, 0.2)
    #             reward_components['patience'] = patience_reward
    #             hold_reward += patience_reward
        
    #     return hold_reward
    
    def _is_invalid_action(self, action):
        """returns bool representing if the action is invalid"""
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]

        if action == 0:  # sell
            if self.shares_held <= 0:
                return True
            sell_price = self.shares_held * current_price
            return sell_price <= self.min_trade_amount
        elif action == 2:  # buy
            if self.balance <= 0 or self.shares_held > 0:
                return True
            max_possible_shares = int(self.balance / (current_price * (1 + self.transaction_fee_pct))) if current_price > 0 else 0
            max_allowed_shares = int(self.initial_balance * self.max_position_size / current_price) if current_price > 0 else 0
            shares_to_buy = min(max_possible_shares, max_allowed_shares)
            return shares_to_buy * current_price <= self.min_trade_amount
        return False
    
    
    def step(self, action):
        """
        Execute one step in the environment based on the agent's action
        Actions: 0=Sell all, 1=Hold, 2=Buy max
        """
        def sell(is_forced_sell: bool=False):
            sell_price = self.shares_held * current_price
            fee = sell_price * self.transaction_fee_pct
            sell_price_adjusted = sell_price - fee
            buy_price = self.entry_price * self.shares_held
            buy_price_fee = buy_price * self.transaction_fee_pct
            buy_price_adjusted = buy_price + buy_price_fee
            did_profit = sell_price_adjusted > buy_price_adjusted
            trade_info = {
                'type': 'END_OF_EPISODE' if is_forced_sell else 'SELL',
                'shares': self.shares_held,
                'price': current_price,
                'sell_amount': sell_price_adjusted,
                'buy_amount': buy_price_adjusted,
                'fee': fee,
                'action': 'FORCED_SELL' if is_forced_sell else action,
                'entry_price': self.entry_price,
                'trade_duration': self.current_trade_duration,
                'did_profit': did_profit
            }
            self.balance += sell_price_adjusted
            self.shares_held = 0
            self.total_trades += 1
            self.total_shares_sold += self.shares_held
            self.consecutive_trades += 1
            self.last_trade_step = self.current_step
            self.entry_price = 0 
            self.max_profit = max(self.max_profit, sell_price_adjusted)
            self.max_loss = min(self.max_loss, sell_price_adjusted)
            if did_profit:
                if self.was_last_trade_profitable:
                    self.consecutive_profits += 1
                self.profitable_trades += 1
                self.total_profit += sell_price_adjusted - buy_price_adjusted
                self.successful_trade_durations.append(self.current_trade_duration)
                if len(self.successful_trade_durations) > 1000:
                    self.successful_trade_durations.pop()
            else:
                if not self.was_last_trade_profitable:
                    self.consecutive_losses += 1
                self.loss_making_trades += 1
                self.total_loss += buy_price_adjusted - sell_price_adjusted
            self.current_trade_duration = 0
            self.trailing_stop_price = 0 
            self.transaction_fee = fee
            return trade_info, did_profit

        def hold():
            self.consecutive_trades = 0
            self.transaction_fee = 0
        def buy():
            max_shares_possible = int(self.balance / (current_price * (1 + self.transaction_fee_pct))) if current_price > 0 else 0
            max_shares_allowed = int(self.initial_balance * self.max_position_size / current_price) if current_price > 0 else 0
            shares_to_buy = min(max_shares_possible, max_shares_allowed)
            buy_amount = shares_to_buy * current_price
            fee = buy_amount * self.transaction_fee_pct
            cost = buy_amount + fee
            trade_info = {
                'type': 'BUY',
                'shares': shares_to_buy,
                'price': current_price,
                'buy_amount': buy_amount + fee,
                'fee': fee,
                'action': action
            }
            self.balance -= cost
            self.shares_held += shares_to_buy
            self.total_shares_bought += shares_to_buy
            self.total_cost += cost
            self.entry_price = current_price
            self.trailing_stop_price = current_price * (1 - self.trailing_stop_threshold)
            self.current_trade_duration += 1
            self.transaction_fee = fee
            return trade_info
    
        reward = 0
        done = False
        trade_info = {}
        did_profit = None
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        invalid_action = self._is_invalid_action(action)
        self.lowest_price = min(self.lowest_price, current_price)
        self.highest_price = max(self.highest_price, current_price)
        
        # force sell all shares at the end of the episode or if current max position size is lower than the minimum trade amount
        if self.current_step >= self.data_nparray.shape[0] - 1 or self._is_out_of_game():
            done = True
            if self.shares_held > 0:
                trade_info, did_profit = sell(True)
        else:
            if invalid_action:
                self.invalid_actions += 1
            else:
                self.action_history.append(action)
                if action == 0:  # sell
                    trade_info, did_profit = sell()
                elif action == 1:  # hold
                    hold()
                elif action == 2:  # buy
                    trade_info = buy()
            self.consecutive_holds = self.consecutive_holds + 1 if (action == 1 and self.shares_held > 0) or invalid_action else 0
            self.current_step += 1
            
        if done and self.mode == 'test':
            trade_info['max_drawdown'] = self._calculate_max_drawdown()
            trade_info['sharpe_ratio'] = self._calculate_sharpe_ratio()
            trade_info['portfolio_values'] = self.portfolio_values
            trade_info['price_history'] = self.price_history
            trade_info['action_history'] = self.action_history
            trade_info['final_balance'] = self.balance
            trade_info['return_rate'] = (self.balance - self.initial_balance) / self.initial_balance
        
        if self.shares_held > 0:
            self.current_trade_duration += 1
        portfolio_value = self.balance + self.shares_held * current_price
        self.highest_portfolio_value_seen_so_far = max(self.highest_portfolio_value_seen_so_far, portfolio_value)
        self.lowest_portfolio_value_seen_so_far = min(self.lowest_portfolio_value_seen_so_far, portfolio_value)
        reward += self._calculate_reward(invalid_action, action, trade_info, done)

        # self.last_action = action
        self.last_portfolio_value = portfolio_value
        self.was_last_trade_profitable = did_profit if did_profit is not None else self.was_last_trade_profitable
        
        return self._get_state(), reward, done, {
            'trade_info': trade_info,
            'reward_components': self.reward_components
        }
    
    
    def _calculate_max_drawdown(self) -> np.float64:
        """
        Calculate the maximum drawdown from peak to trough
        """
        values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(values)
        drawdown = (running_max - values) / running_max
        max_drawdown = np.max(drawdown)
        return max_drawdown
    
    
    def _calculate_sharpe_ratio(self) -> np.float64:
        """
        Calculate the Sharpe ratio of the portfolio
        """
        values = np.array(self.portfolio_values)
        daily_returns = np.diff(values) / values[:-1]
        excess_returns = daily_returns - self.risk_free_rate
        if np.std(excess_returns) == 0:
            return 0
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio
    
    
    def get_branch_sizes(self):
        if self.use_hierarchical:
            return {
                'stock_data_window_size': self.window_size,
                'stock_data_feature_size': len(self.state_scaler.features),
                'portfolio_metrics_size': 8,
                'performance_metrics_size': 15,
                'risk_metrics_size': 5,
                'market_state_metrics_size': 15,
                'position_management_metrics_size': 7,
                'trading_behavior_metrics_size': 7,
                'temporal_metrics_size': 2,
                'temporal_metrics_types_count': 3,
                # 'temporal_metrics_types_count': 5,
                'action_size': 3
            }
        return {
            'stock_data_window_size': self.window_size,
            'stock_data_feature_size': len(self.state_scaler.features),
            'portfolio_metrics_size': 32,
            'market_state_metrics_size': 15,
            'constraint_metrics_size': 10,
            'action_size': 3
        }