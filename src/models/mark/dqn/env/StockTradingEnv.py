import numpy as np
import pandas as pd
import random
import torch
from numpy.typing import NDArray

from src.config.config import (
    WINDOW_SIZE,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT
)
from src.preprocessing.data_processor import RollingWindowFeatureProcessor

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
        initial_balance: int=INITIAL_BALANCE, 
        transaction_fee: float=TRANSACTION_FEE_PERCENT, 
        window_size: int=WINDOW_SIZE,
        mode: str='train'  # 'train', 'validation', or 'test'
    ):  
        # TODO might be able to remove self.data
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
        
        # trading constraints
        self.min_trade_interval: int = 2     # minimum interval between trades
        self.max_position_size: float = 0.7  # maximum position size (how much capitol can be used per trade)
        self.min_trade_amount: int = 300     # minimum amount required to make a trade
        self.critical_loss_value: float = 0.5 * initial_balance
        
        # reward settings
        self.profit_reward_weight: int = 75                         # profit reward weight (used to scale rewards for profitable trades)
        self.loss_penalty_weight: int = 60                          # loss penalty weight (used to scale penalty for loss)
        self.trade_reward: float = 0.15                             # reward for trade execution
        self.successful_trade_reward: float = 0.5                   # reward for successful trade
        self.patience_reward: float = 0.05                          # reward for waiting for better opportunities
        self.efficient_capital_usage_reward: float = 0.1            # reward for efficient capital usage
        self.invalid_action_penalty: float = -1.5                   # penalty for taking invalid action
        self.stop_loss_penalty: float = -3000                       # penalty for reaching stop loss thresholdd
        self.critical_loss_penalty: float = -8000                   # penalty for reaching critical loss threshold
        self.small_hold_time_penalty: float = -0.0003               # penalty for holding 
        self.consecutive_hold_penalty_base: float = -0.01           # penalty for consecutive holding
        self.consecutive_hold_penalty_max: float = -0.1             # penalty for consecutive holding maximum
        self.exceed_max_profit_threshold_penalty: float = -0.012    # penalty for holding past max profit threshold
        self.exceed_profit_threshold_penalty: float = -0.006        # penalty for holding past profit threshold
        self.exceed_max_loss_threshold_penalty: float = -0.1        # penalty for holding past max loss threshold
        self.exceed_loss_threshold_penalty: float = -0.05           # penalty for holding past loss threshold
        self.price_movement_alignment_bonus: float = 0.3            # bonus when action aligns with price movement
        self.quick_profit_taking_bonus: float = 0.2                 # bonus for taking profits quickly

        # profit/loss threshold settings
        self.profit_threshold: float = 0.012        # profit-taking threshold
        self.loss_threshold: float = -0.008         # loss-cutting threshold
        self.trailing_stop_threshold: float = 0.007 # trailing stop value
        self.max_profit_threshold: float = 0.025    # maximum profit-taking threshold
        self.max_loss_threshold: float = -0.025     # maximum loss threshold

        # additional profitability-related settings
        self.profit_taking_levels = [0.01, 0.015, 0.02, 0.03]   # tiered profit-taking levels
        self.profit_taking_weights = [1.0, 1.3, 1.6, 2.0]       # reward weights for each profit-taking level
        self.volatility_threshold: float = 0.018                # volatility threshold (used to determine whether to adjust behavior during periods of high volatility)
        self.trend_following_weight: float = 1.5                # trend-following reward weight (encourages the agent to follow the trend by increasing rewards when aligned with it)
        
        # market regime settings
        self.market_regime_weights = {
            'trending_up': 1.2,       # bonus for buying in uptrends
            'trending_down': 1.2,     # bonus for selling in downtrends
            'ranging': 0.8,           # reduced reward in ranging markets
            'high_volatility': 0.7    # reduced reward in high volatility
        }
        self.position_sizing_factor: float = 0.1  # reward component for optimal position sizing
        
        # state variables
        self.current_step: int = 0
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
        self.last_action = None
        self.last_trade_step: int = -1
        self.trade_history = []
        self.entry_price: int = 0
        self.max_profit: int = 0
        self.max_loss: int = 0
        self.current_trade_duration: int = 0
        self.trailing_stop_price: int = 0
        self.position_open: bool = False
        self.current_drawdown: float = 0
        self.max_drawdown: float = 0
        self.invalid_actions: int = 0
        self.highest_price_since_buy: float = 0
        self.lowest_price_since_buy: float = 0
        
        
        # market state
        self.market_regime: str = 'unknown'
        self.market_volatility: float = 0
        
        # portfolio metrics tracking
        self.portfolio_values = [self.initial_balance]
        self.action_history = []
        self.price_history = []
        self.reward_components = []
        
        # attributes for calculating state_size
        # TODO make this more streamline
        self.feature_extracted_column_count = self.feature_processor.feature_processor.n_components
        self.portfolio_info_count = 21
        self.market_info_count = 7
        self.constraint_info_count = 6
        
        self.reset()
        
        
    def reset(self) -> NDArray:
        # start at window_size to ensure enough historical data for the first state's features
        self.current_step = self.window_size 
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
        self.last_action = None
        self.last_trade_step = -1
        self.trade_history = []
        self.entry_price = 0
        self.max_profit = 0
        self.max_loss = 0
        self.current_trade_duration = 0
        self.trailing_stop_price = 0
        self.position_open = False
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.invalid_actions = 0
        self.highest_price_since_buy = 0
        self.lowest_price_since_buy = 0
        self.market_regime = 'unknown'
        self.market_volatility = 0
        self.portfolio_values = [self.initial_balance]
        self.action_history = []
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
        features = self.data_nparray[start_idx:end_idx]
        # rolling window scaling: Fit and transform on the features of the current window
        processed_features = self.feature_processor.get_state(features)
        self._feature_cache[current_idx] = processed_features
        return processed_features


    def _get_state(self) -> NDArray:
        """
        Construct the current state 
        """
        # normalized features
        normalized_features_flattened = self._get_features(self.current_step)
        
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        portfolio_value = self.balance + self.shares_held * current_price
        
        # market regime and volatility
        self.market_regime, self.market_volatility = self._detect_market_regime()
        
        # calculate current drawdown
        peak_value = max(self.portfolio_values)
        self.current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Enhanced portfolio information
        normalized_portfolio_value = portfolio_value / self.initial_balance if self.initial_balance > 0 else 0
        balance_ratio = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        shares_value_ratio = (self.shares_held * current_price) / self.initial_balance if self.initial_balance > 0 else 0
        
        # position metrics
        is_holding_shares = float(self.shares_held > 0)
        normalized_shares_held = self.shares_held / (self.initial_balance / current_price) if current_price > 0 else 0
        position_utilization = (self.shares_held * current_price) / (self.initial_balance * self.max_position_size) if self.initial_balance > 0 else 0
        
        # Calculate profit/loss from current position
        avg_buy_price = self.total_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0
        position_pl = (current_price - avg_buy_price) * self.shares_held if self.shares_held > 0 else 0
        position_pl_ratio = position_pl / (self.total_cost if self.total_cost > 0 else 1)
        normalized_position_pl = position_pl / self.initial_balance if self.initial_balance > 0 else 0

        # trading performance metrics
        win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0
        profit_factor = 1.0  # Default value
        if self.loss_making_trades > 0:
            profit_factor = self.profitable_trades / self.loss_making_trades
            
        # time-based states
        time_in_position = self.current_trade_duration / self.steps_per_episode if self.position_open else 0
        time_since_last_trade = np.clip((self.current_step - self.last_trade_step) / self.steps_per_episode, 0, 1)
        
        # recent price movement (short-term momentum)
        if self.current_step > 5:
            recent_price_change = (current_price / self.data_nparray[self.current_step-5, self.close_prices_idx]) - 1
        else:
            recent_price_change = 0
            
        # trend alignment indicators
        trend_direction = 0
        if self.market_regime == 'trending_up':
            trend_direction = 1
        elif self.market_regime == 'trending_down':
            trend_direction = -1
            
        highest_price_since_buy_and_entry_price_ratio = self.highest_price_since_buy / self.entry_price if self.entry_price > 0 else 0
        highest_price_since_buy_and_current_price_ratio = self.highest_price_since_buy / current_price if current_price > 0 else 0
        lowest_price_since_buy_and_entry_price_ratio = self.lowest_price_since_buy / self.entry_price if self.entry_price > 0 else 0
        lowest_price_since_buy_and_current_price_ratio = self.lowest_price_since_buy / current_price if current_price > 0 else 0

        normalized_proximity_to_critical_loss = (portfolio_value - self.critical_loss_penalty) / self.initial_balance

        ###########################################################################################
        ######################################### READ ME #########################################
        ###########################################################################################
        # if portfolio_info, market_info, or constraint_info length changes update initial values of
        # self.portfolio_info_count, self.market_info_count, self.constraint_info_count respectively
        
        # portfolio info states
        portfolio_info = np.array([
            normalized_portfolio_value,                         # normalized portfolio value
            balance_ratio,                                      # ratio of current and initial balance
            shares_value_ratio,                                 # ratio of shares value to initial balance
            is_holding_shares,                                  # boolean if holding shares
            normalized_position_pl,                             # normalized profit/loss on current position
            position_pl_ratio,                                  # profit/loss as percentage of position cost
            normalized_shares_held,                             # normalized number of shares held
            position_utilization,                               # how much of max position size is utilized
            win_rate,                                           # win rate of trades
            profit_factor,                                      # ratio of profitable to losing trades
            self.consecutive_profits / 10,                      # normalized consecutive profitable trades
            self.consecutive_losses / 10,                       # normalized consecutive losing trades
            time_in_position,                                   # normalized time in current position
            self.current_drawdown,                              # current drawdown
            self.max_drawdown,                                  # maximum drawdown
            self.invalid_actions / self.steps_per_episode,      # normalized invalid acounts count
            highest_price_since_buy_and_entry_price_ratio,      # ratio of highest price since buy and entry price
            highest_price_since_buy_and_current_price_ratio,    # ratio of highest price since buy and current price
            lowest_price_since_buy_and_entry_price_ratio,       # ratio of lowest price since buy and entry price
            lowest_price_since_buy_and_current_price_ratio,     # ratio of lowest price since buy and current price
            normalized_proximity_to_critical_loss
        ], dtype=np.float32)
        
        # market info states
        market_info = np.array([
            float(self.market_regime == 'trending_up'),     # is market trending up
            float(self.market_regime == 'trending_down'),   # is market trending down
            float(self.market_regime == 'ranging'),         # is market ranging
            float(self.market_regime == 'high_volatility'), # is market highly volatile
            self.market_volatility,                         # market volatility
            recent_price_change,                            # recent price change
            trend_direction                                 # trend direction indicator
        ], dtype=np.float32)
        
        # trading constraints and behavioral information
        constraint_info = np.array([
            self.consecutive_trades / (self.steps_per_episode / 2),                 # normalized consecutive trades
            time_since_last_trade,                                                  # normalized time since last trade
            self.consecutive_holds / self.steps_per_episode,                        # normalized consecutive holds 
            float(self.last_action) / 2 if self.last_action is not None else 0.5,   # normalized last action (-0.5, 0, 0.5)
            len(self.trade_history) / (self.steps_per_episode / 2),                 # normalized trade history length
            self.total_trades / (self.steps_per_episode / 2)                        # normalized total trades
        ], dtype=np.float32)

        # track portfolio value and price history
        self.price_history.append(current_price)
        self.portfolio_values.append(portfolio_value)
            
        return np.concatenate((
            normalized_features_flattened, 
            portfolio_info, 
            market_info,
            constraint_info
        )).astype(np.float32)
    
    
    def _calculate_reward(self, action, trade_info, done):
        """
        Calculate the reward for the current step
        """
        reward_components = {}
        total_reward = 0.0
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]
        portfolio_value = self.balance + self.shares_held * current_price
        # if self._is_invalid_action(action):
        #     reward_components['invalid_action'] = self.invalid_action_penalty
        #     total_reward += reward_components['invalid_action']
        # elif action != 1:
        #     portfolio_value = self.balance + self.shares_held * current_price
        #     final_profit = (portfolio_value - self.initial_balance) / self.initial_balance
        #     reward_components['profit'] = final_profit
        #     total_reward += reward_components['profit']
        # else:
        #     reward_components['hold_penalty'] = self.small_hold_time_penalty
        #     total_reward += reward_components['hold_penalty']
            
        #     if self.shares_held > 0 and self.position_open:
        #         # Progressive holding penalty based on consecutive holds
        #         if self.market_regime != 'ranging':
        #             hold_penalty = self.consecutive_hold_penalty_base * (1 + 0.02 * self.consecutive_holds)
        #             hold_penalty = min(hold_penalty, self.consecutive_hold_penalty_max)  # Cap the penalty
        #             reward_components['progressive_hold_penalty'] = hold_penalty
        #             total_reward += hold_penalty
                
        #         # Position profit/loss evaluation
        #         if self.entry_price > 0:
        #             current_position_pct = (current_price - self.entry_price) / self.entry_price
                    
        #             # Penalties for holding beyond thresholds
        #             if current_position_pct > self.max_profit_threshold:
        #                 reward_components['exceed_max_profit'] = self.exceed_max_profit_threshold_penalty
        #                 total_reward += reward_components['exceed_max_profit']
        #             elif current_position_pct > self.profit_threshold:
        #                 reward_components['exceed_profit'] = self.exceed_profit_threshold_penalty
        #                 total_reward += reward_components['exceed_profit']
        #             elif current_position_pct < self.max_loss_threshold:
        #                 reward_components['exceed_max_loss'] = self.exceed_max_loss_threshold_penalty
        #                 total_reward += reward_components['exceed_max_loss']
        #             elif current_position_pct < self.loss_threshold:
        #                 reward_components['exceed_loss'] = self.exceed_loss_threshold_penalty
        #                 total_reward += reward_components['exceed_loss']
                
        #     # else:
        #         # Patience reward for waiting while having no position (only if we have enough balance)
        #         if self.balance > self.initial_balance * 0.5 and self.consecutive_holds > 5:
        #             patience_reward = min(self.patience_reward * (self.consecutive_holds / 20), self.patience_reward * 2)
        #             if self.market_regime == 'trending_down':
        #                 patience_reward *= 1.3
        #             reward_components['patience'] = patience_reward
        #             total_reward += patience_reward 
        
        # end of episode state additional rewards/penalties
        if done:
            final_profit = (portfolio_value - self.initial_balance) / self.initial_balance
            
            # Reward/penalty based on final performance
            if final_profit > 0:
                # Stronger reward for ending with profit
                final_reward = final_profit * self.profit_reward_weight * 3
                reward_components['final_profit'] = final_reward
                total_reward += final_reward
                
                # Additional reward based on consistency
                if self.profitable_trades > self.loss_making_trades * 1.5:  # At least 60% win rate
                    consistency_reward = final_profit * 0.5
                    reward_components['consistency'] = consistency_reward
                    total_reward += consistency_reward
            else:
                # Penalty for ending with a loss
                final_penalty = final_profit * self.loss_penalty_weight * 2
                reward_components['final_loss'] = final_penalty
                total_reward += final_penalty
                
                # Extra penalty for severe losses
                if final_profit < -0.3:
                    severe_loss_penalty = -5 * abs(final_profit)
                    reward_components['severe_loss'] = severe_loss_penalty
                    total_reward += severe_loss_penalty

            # Drawdown penalty
            if self.max_drawdown > 0.2:  # More than 20% drawdown
                drawdown_penalty = -self.max_drawdown * 5
                reward_components['max_drawdown'] = drawdown_penalty
                total_reward += drawdown_penalty
                
            # Capital efficiency reward
            trade_frequency = self.total_trades / (self.steps_per_episode / 2)
            if trade_frequency > 0.1 and final_profit > 0:  # Reward active trading if profitable
                capital_efficiency = self.efficient_capital_usage_reward * trade_frequency * 10
                reward_components['capital_efficiency'] = capital_efficiency
                total_reward += capital_efficiency
            
            if self.invalid_actions > 0:
                reward_components['invalid_actions_scaling_factor'] = 1.0 / (1 + (self.invalid_actions * 0.1))
                total_reward *= reward_components['invalid_actions_scaling_factor'] 
        else:
            # Base portfolio change component
            portfolio_change_pct = (portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value if self.last_portfolio_value != 0 else self.initial_balance)
            reward_components['portfolio_change'] = portfolio_change_pct * 2  # Double weight on actual portfolio performance
            total_reward += reward_components['portfolio_change']
            
            # Overall profitability component
            overall_profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance
            reward_components['overall_profit'] = overall_profit_pct
            total_reward += reward_components['overall_profit']
            
            # Invalid action penalty
            if self._is_invalid_action(action):
                reward_components['invalid_action'] = self.invalid_action_penalty
                total_reward += reward_components['invalid_action']
            else:
                # Handle specific trade actions
                if trade_info.get('action') == 'FORCED_SELL':
                    # Handle forced sells (stop loss, trailing stop, critical loss)
                    match(trade_info.get('type')):
                        case 'STOP_LOSS_SELL':
                            reward_components['stop_loss'] = self.stop_loss_penalty * 0.5  # Reduced penalty for proper stop loss
                            total_reward += reward_components['stop_loss']
                        case 'TRAILING_STOP_SELL':
                            # Smaller penalty for trailing stop as it's protecting profits
                            reward_components['trailing_stop'] = self.stop_loss_penalty * 0.3
                            total_reward += reward_components['trailing_stop']
                        case 'CRITICAL_LOSS_SELL':
                            # Full penalty for hitting critical loss level
                            reward_components['critical_loss'] = self.critical_loss_penalty
                            total_reward += reward_components['critical_loss']
                        case _:
                            print('Unknown trade info:', trade_info.get('type'))
                            
                elif trade_info.get('type') == 'BUY':
                    # Reward/penalty based on market regime alignment
                    if self.market_regime == 'trending_up':
                        # Bonus for buying in uptrend
                        reward_components['trend_alignment'] = self.trade_reward * self.market_regime_weights['trending_up']
                        total_reward += reward_components['trend_alignment']
                    elif self.market_regime == 'trending_down':
                        # Penalty for buying in downtrend
                        reward_components['trend_alignment'] = -self.trade_reward * 0.5
                        total_reward += reward_components['trend_alignment']
                    
                    # Base trade reward
                    reward_components['trade_execution'] = self.trade_reward
                    total_reward += reward_components['trade_execution']
                    
                    # Position sizing component
                    optimal_position_size = self.initial_balance * self.max_position_size
                    actual_position_size = self.shares_held * current_price
                    position_sizing_ratio = min(actual_position_size / optimal_position_size, 1.0) if optimal_position_size > 0 else 0
                    reward_components['position_sizing'] = position_sizing_ratio * self.position_sizing_factor
                    total_reward += reward_components['position_sizing']
                    
                elif trade_info.get('type') == 'SELL':
                    # Calculate profit/loss from this trade
                    if 'amount' in trade_info:
                        profit_amount = trade_info['amount']
                        profit_pct = profit_amount / (self.entry_price * self.shares_held) if self.entry_price > 0 and self.shares_held > 0 else 0
                        
                        # Reward based on profit
                        if profit_pct > 0:
                            # Scale reward based on profit percentage
                            profit_reward = profit_pct * self.profit_reward_weight
                            
                            # Additional reward for quick profitable trades
                            if self.current_trade_duration < self.steps_per_episode * 0.3:  # Less than 30% of possible trade steps
                                profit_reward *= 1.2  # 20% bonus for quick profits
                                reward_components['quick_profit'] = self.quick_profit_taking_bonus
                                total_reward += reward_components['quick_profit']
                                
                            reward_components['profit_reward'] = profit_reward
                            total_reward += reward_components['profit_reward']
                            
                            # Additional tiered reward based on profit levels
                            for level, weight in zip(self.profit_taking_levels, self.profit_taking_weights):
                                if profit_pct > level:
                                    tier_reward = profit_pct * weight
                                    reward_components[f'profit_tier_{level}'] = tier_reward
                                    total_reward += tier_reward
                        else:
                            # Penalty based on loss
                            loss_penalty = profit_pct * self.loss_penalty_weight
                            reward_components['loss_penalty'] = loss_penalty
                            total_reward += loss_penalty
                    
                    # Trend alignment for selling
                    if self.market_regime == 'trending_down':
                        # Bonus for selling in downtrend
                        reward_components['trend_alignment'] = self.trade_reward * self.market_regime_weights['trending_down']
                        total_reward += reward_components['trend_alignment']
                        
                    # Base trade reward
                    reward_components['trade_execution'] = self.trade_reward
                    total_reward += reward_components['trade_execution']
                    
                else:  # HOLD action
                    # Small continuous penalty for holding to encourage decisive action
                    reward_components['hold_penalty'] = self.small_hold_time_penalty
                    total_reward += reward_components['hold_penalty']
                    
                    if self.shares_held > 0 and self.position_open:
                        # Progressive holding penalty based on consecutive holds
                        if self.market_regime != 'ranging':
                            hold_penalty = self.consecutive_hold_penalty_base * (1 + 0.02 * self.consecutive_holds)
                            hold_penalty = min(hold_penalty, self.consecutive_hold_penalty_max)  # Cap the penalty
                            reward_components['progressive_hold_penalty'] = hold_penalty
                            total_reward += hold_penalty
                        
                        # Position profit/loss evaluation
                        if self.entry_price > 0:
                            current_position_pct = (current_price - self.entry_price) / self.entry_price
                            
                            # Penalties for holding beyond thresholds
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
                        # Patience reward for waiting while having no position (only if we have enough balance)
                        if self.balance > self.initial_balance * 0.5 and self.consecutive_holds > 5:
                            patience_reward = min(self.patience_reward * (self.consecutive_holds / 20), self.patience_reward * 2)
                            if self.market_regime == 'trending_down':
                                patience_reward *= 1.3
                            reward_components['patience'] = patience_reward
                            total_reward += patience_reward  

        # Store reward components for debugging/analysis
        self.reward_components.append(reward_components)
        return total_reward

    
    def _is_invalid_action(self, action):
        """returns bool representing if the action is invalid"""
        current_price = self.data_nparray[self.current_step, self.close_prices_idx]

        if action == 0:  # sell
            if self.shares_held <= 0:
                return True
            sell_price = self.shares_held * current_price
            return sell_price <= self.min_trade_amount
        elif action == 2:  # buy
            # cannot buy if no balance or already in a position (prevent averaging down and for simple trading cycle)
            # check if the amount to buy is below the minimum trade amount
            if self.balance <= 0 or self.position_open:
                if self.balance <= 0:
                    print('NO MONEY')
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
        did_profit = None
        
        # check if minimum trading interval has passed since last trade
        can_trade = (self.current_step - self.last_trade_step) >= self.min_trade_interval

        # force sell all shares at the end of the episode
        if self.current_step >= self.data_nparray.shape[0] - 1:
            done = True
            if self.shares_held > 0:
                sell_price = self.shares_held * current_price
                fee = sell_price * self.transaction_fee
                sell_price_adjusted = sell_price - fee
                buy_price = self.entry_price * self.shares_held
                buy_price_fee = buy_price * self.transaction_fee
                buy_price_adjusted = buy_price + buy_price_fee
                did_profit = sell_price_adjusted > buy_price_adjusted
                trade_info = {
                    'type': 'END_OF_EPISODE',
                    'shares': self.shares_held,
                    'price': current_price,
                    'amount': sell_price,
                    'fee': fee,
                    'action': 'FORCED_SELL',
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
                else:
                    if not self.was_last_trade_profitable:
                        self.consecutive_losses += 1
                    self.loss_making_trades += 1
                self.current_trade_duration = 0
                self.trailing_stop_price = 0
                self.position_open = False
        else:
            self.action_history.append(action)
            invalid_action = self._is_invalid_action(action)
            if invalid_action:
                reward += self._calculate_reward(action, trade_info, False)
                self.invalid_actions += 1
            else:
                if action == 0:  # sell
                    if not can_trade:
                        reward -= 0.1
                    sell_price = self.shares_held * current_price
                    fee = sell_price * self.transaction_fee
                    sell_price_adjusted = sell_price - fee
                    buy_price = self.entry_price * self.shares_held
                    buy_price_fee = buy_price * self.transaction_fee
                    buy_price_adjusted = buy_price + buy_price_fee
                    did_profit = sell_price_adjusted > buy_price_adjusted
                    trade_info = {
                        'type': 'SELL',
                        'shares': self.shares_held,
                        'price': current_price,
                        'amount': sell_price,
                        'fee': fee,
                        'action': action,
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
                    else:
                        if not self.was_last_trade_profitable:
                            self.consecutive_losses += 1
                        self.loss_making_trades += 1
                    self.current_trade_duration = 0
                    self.trailing_stop_price = 0 
                    self.position_open = False 
                elif action == 1:  # hold
                    self.consecutive_trades = 0
                    if self.shares_held > 0 and self.position_open:
                        self.current_trade_duration += 1
                elif action == 2:  # buy
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
                    self.total_shares_bought += shares_to_buy
                    self.total_cost += cost
                    self.entry_price = current_price
                    self.trailing_stop_price = current_price * (1 - self.trailing_stop_threshold)
                    self.position_open = True
            
        if done:
            if self.mode == 'test':
                trade_info['max_drawdown'] = self._calculate_max_drawdown(self.portfolio_values)
                trade_info['sharpe_ratio'] = self._calculate_sharpe_ratio(self.portfolio_values)
                trade_info['portfolio_values'] = self.portfolio_values
                trade_info['price_history'] = self.price_history
                trade_info['action_history'] = self.action_history
                trade_info['final_balance'] = self.balance
                trade_info['return_rate'] = (self.balance - self.initial_balance) / self.initial_balance
        
        if not done:
            self.current_step += 1
        self.consecutive_holds = self.consecutive_holds + 1 if action == 1 else 0
        reward += self._calculate_reward(action, trade_info, done)
            
        if trade_info:
            self.trade_history.append(trade_info)

        self.last_action = action
        self.last_portfolio_value = self.balance + self.shares_held * current_price
        self.was_last_trade_profitable = did_profit if did_profit is not None else self.was_last_trade_profitable
        
        return self._get_state(), reward, done, {
            'trade_info': trade_info,
            'reward_components': self.reward_components
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