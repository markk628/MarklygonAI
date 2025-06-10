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
    num_portfolio_features: int = 12
    num_features: int = len(STOCK_FEATURES_V2) + 12  # Stock features + portfolio features
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
    epsilon_end: float = 0.12 
    epsilon_decay: float = 100000  
    
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
    
    # Balanced exploration for smart active trading  
    config.epsilon_start = 1.0
    config.epsilon_end = 0.10  # Moderate final epsilon for strategic decisions
    config.epsilon_decay = 100000  # Balanced decay
    
    # Adjusted learning parameters for better exploration
    config.learning_rate = 0.0003  # Slightly higher learning rate
    config.tau = 0.01  # Faster target network updates
    
    # More frequent updates for faster learning
    config.update_frequency = max(1, config.update_frequency // 2)  # 2x more frequent updates
    
    # Adjusted buffer parameters
    config.alpha = 0.7  # Higher prioritization
    config.beta_start = 0.5  # Higher importance sampling
    
    print("🚀 Created SMART AGGRESSIVE trading configuration:")
    print(f"   • Balanced final epsilon: {config.epsilon_end}")
    print(f"   • Strategic epsilon decay: {config.epsilon_decay}")
    print(f"   • Higher learning rate: {config.learning_rate}")
    print(f"   • More frequent updates: every {config.update_frequency} steps")
    print(f"   • Technical analysis-based reward system")
    print(f"   • Smart invalid action penalties")
    print(f"   • Multi-indicator decision making (RSI, MACD, Momentum, Volume, etc.)")
    
    return config 