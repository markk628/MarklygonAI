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
    learning_rate: float = 0.0004982288426094389
    
    # Training parameters
    batch_size: int = 64
    gamma: float = 0.9801847140197922
    tau: float = 0.0011031557332547512
    update_frequency: int = 1
    target_update_frequency: int = UPDATE_TARGET_EVERY
    
    # Experience replay
    buffer_size: int = REPLAY_BUFFER_SIZE
    
    # Exploration
    epsilon_start: float = 0.8963735840460665
    epsilon_end: float = 0.014450212613536453  # Lower minimum exploration for trading
    epsilon_decay: float = 1295  # Much faster decay to reduce random trading losses
    
    # Prioritized replay
    use_prioritized_replay: bool = True
    alpha: float = 0.7743174729030339
    beta_start: float = 0.48623984012473076
    beta_end: float = 0.9969314272861748
    per_epsilon: float = 0.004641094476076944
    
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
    min_profit_threshold: float = 0.0045221048969825576  # Minimum 1.5% expected profit to trade (3x more aggressive)
    
    # Optimizable reward parameters (can be tuned via Optuna)
    portfolio_scaling: float = 0.11509182887901824  # Scaling factor for portfolio value changes
    invalid_penalty: float = 0.07906291713801883   # Penalty for invalid actions
    
    def __post_init__(self):
        if self.cnn_scales is None:
            self.cnn_scales = [3, 5, 7]