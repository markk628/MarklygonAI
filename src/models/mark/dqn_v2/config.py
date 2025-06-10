from enum import Enum
from dataclasses import dataclass
from src.config.config import (
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    MAX_POSITION_SIZE,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    UPDATE_TARGET_EVERY,
    STOCK_FEATURES_V2,
    TRAIN_INTERVAL
)


class ArchitectureType(Enum):
    ORIGINAL = "original"
    IMPROVED = "improved"
    HYBRID = "hybrid"


@dataclass
class TradingConfig:
    """Configuration for the DQN agent"""
    # Environment parameters
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    window_size: int = WINDOW_SIZE
    num_stock_features: int = len(STOCK_FEATURES_V2)
    num_portfolio_features: int = 13  # Updated for new invalid_action_rate feature
    num_features: int = len(STOCK_FEATURES_V2) + 13  # Stock features + portfolio features
    num_actions: int = 3  # Hold, Buy, Sell
    max_position_size: float = MAX_POSITION_SIZE
    
    # Network architecture selection
    architecture_type: ArchitectureType = ArchitectureType.IMPROVED
    
    # Network parameters
    hidden_size: int = 512
    learning_rate: float = 0.0001
    
    # Training parameters
    batch_size: int = BATCH_SIZE
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = TRAIN_INTERVAL
    target_update_frequency: int = UPDATE_TARGET_EVERY
    
    # Experience replay
    buffer_size: int = REPLAY_BUFFER_SIZE
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01  # Lower minimum exploration for trading
    epsilon_decay: float = 2000  # Much faster decay to reduce random trading losses
    
    # Prioritized replay
    use_prioritized_replay: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    per_epsilon: float = 0.001
    
    # Double and Dueling DQN
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    
    # Enhanced architecture options (for improved/hybrid)
    use_attention: bool = True
    use_residual_connections: bool = True
    transformer_layers: int = 2
    cnn_scales: list = None  # [3, 5, 7] if None
    
    # Portfolio state normalization
    use_portfolio_normalization: bool = True
    portfolio_warmup_episodes: int = 50
    portfolio_update_frequency: int = 100
    
    # Trading-specific parameters (from dqn_v6 improvements)
    min_profit_threshold: float = 0.015  # Minimum 1.5% expected profit to trade (3x more aggressive)
    patience_bonus_rate: float = 0.0005  # Bonus for holding positions (5x stronger)
    trading_frequency_penalty: float = 0.008  # Penalty for excessive trading (4x stronger)
    
    # Enhanced trading discipline parameters
    post_trade_cooldown_penalty: float = 0.012  # Penalty for trading too soon after previous trade
    reflection_bonus_rate: float = 0.0003  # Bonus for staying in cash after losing trades
    min_hold_time_steps: int = 5  # Minimum steps to hold position before selling (thoughtful exits)
    
    # Optimizable reward parameters (can be tuned via Optuna)
    portfolio_scaling: float = 0.1  # Scaling factor for portfolio value changes
    invalid_penalty: float = 0.1   # Penalty for invalid actions
    
    def __post_init__(self):
        if self.cnn_scales is None:
            self.cnn_scales = [3, 5, 7]


def create_aggressive_trading_config(base_config: TradingConfig = None) -> TradingConfig:
    """
    Create a more aggressive trading configuration to encourage higher trading frequency
    and better performance. Use this if your model is too conservative.
    
    Args:
        base_config: Base configuration to modify, or None to use default
        
    Returns:
        Enhanced TradingConfig for more active trading
    """
    if base_config is None:
        config = TradingConfig()
    else:
        # Copy the base config
        import copy
        config = copy.deepcopy(base_config)
    
    # Faster exploration decay for efficient trading  
    config.epsilon_start = 1.0
    config.epsilon_end = 0.01  # Lower minimum exploration for trading
    config.epsilon_decay = 2000  # Much faster decay to reduce random trading losses
    
    # Adjusted learning parameters for better exploration
    config.learning_rate = 0.0003  # Slightly higher learning rate
    config.tau = 0.01  # Faster target network updates
    
    # More frequent updates for faster learning
    config.update_frequency = max(1, config.update_frequency // 2)  # 2x more frequent updates
    
    # Adjusted buffer parameters
    config.alpha = 0.7  # Higher prioritization
    config.beta_start = 0.5  # Higher importance sampling
    
    print("🚀 Created ENHANCED DISCIPLINED TRADING configuration:")
    print(f"   • Fast exploration decay: {config.epsilon_decay} (vs 100,000)")
    print(f"   • Low final epsilon: {config.epsilon_end} (reduced random trading)")
    print(f"   • Higher learning rate: {config.learning_rate}")
    print(f"   • More frequent updates: every {config.update_frequency} steps")
    print(f"   • Minimum profit threshold: {config.min_profit_threshold:.1%} (3x higher)")
    print(f"   • Patience bonus system: {config.patience_bonus_rate} (5x stronger)")
    print(f"   • Trading frequency penalties: {config.trading_frequency_penalty} (4x stronger)")
    print(f"   • Post-trade cooldown penalty: {config.post_trade_cooldown_penalty}")
    print(f"   • Reflection bonus rate: {config.reflection_bonus_rate}")
    print(f"   • Minimum hold time: {config.min_hold_time_steps} steps")
    print(f"   • Portfolio scaling: {config.portfolio_scaling} (optimizable)")
    print(f"   • Invalid penalty: {config.invalid_penalty} (optimizable)")
    print(f"   • 🎯 DESIGNED TO STOP BUY-AFTER-SELL BEHAVIOR!")
    print(f"   • 🔧 Use dqn_v5_optimize_rewards.py to tune reward parameters!")
    
    return config 