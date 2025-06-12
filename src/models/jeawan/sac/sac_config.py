"""
SAC Configuration Classes
========================

Centralized configuration for all SAC components to avoid duplication
and provide clean separation of concerns.
"""

from enum import Enum
from src.config.config import (
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    MAX_POSITION_SIZE,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    STOCK_FEATURES_V2,
    MINUTES_PER_TRADING_DAY
)


class TradingMode(Enum):
    """Trading modes for different phases"""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class EnvironmentType(Enum):
    """Available trading environment types"""
    BASIC = 'basic'                    # Simple buy-sell cycles (SAC v1)
    WEIGHTED_AVERAGE = 'weighted_avg'  # Weighted average cost basis (SAC v2)
    LOT_BASED = 'lot_based'           # Individual lot tracking (SAC v3)


class NetworkType(Enum):
    """Available network architectures"""
    ORIGINAL = 'original'      # Original complex networks
    SIMPLIFIED = 'simplified'  # Simplified networks (recommended)


class SACConfig:
    """Configuration for SAC agent and environments"""
    
    # Environment parameters
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    window_size: int = WINDOW_SIZE
    num_stock_features: int = len(STOCK_FEATURES_V2)
    num_portfolio_features: int = 20  # Updated from 13 to 20 for enhanced action guidance
    num_features: int = len(STOCK_FEATURES_V2) + 20  # Updated total features
    max_position_size: float = MAX_POSITION_SIZE
    minutes_per_day: int = MINUTES_PER_TRADING_DAY
    
    # Network parameters
    actor_hidden_size: int = 512
    critic_hidden_size: int = 512
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    
    # Training parameters
    batch_size: int = BATCH_SIZE
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    update_frequency: int = 1
    
    # Experience replay
    buffer_size: int = REPLAY_BUFFER_SIZE
    
    # SAC specific parameters
    target_entropy: float = -0.5  # Less conservative than -1.0 for more active trading
    alpha_auto_tune: bool = True  # Re-enabled for proper SAC learning
    initial_alpha: float = 0.1  # Starting value, will be auto-tuned
    
    # Portfolio state normalization
    use_portfolio_normalization: bool = True
    portfolio_warmup_episodes: int = 50
    portfolio_update_frequency: int = 100
    
    # Trading parameters
    min_trade_amount: float = 0.001  # Minimum 0.1% position size for trades
    
    # Reward parameters (balanced for SAC)
    portfolio_scaling: float = 0.1  # Balanced scaling - enough incentive for trading, stable for neural networks
    invalid_penalty: float = 0.05  # Reduced penalty for invalid actions
    
    # PER parameters
    per_alpha: float = 0.6  # Prioritization strength
    per_beta_start: float = 0.4  # Initial importance sampling
    per_beta_end: float = 1.0  # Final importance sampling
    per_epsilon: float = 0.001  # Small constant for numerical stability
    per_beta_annealing_steps: int = 100000  # Steps to anneal beta
    
    # Learning Rate Scheduler parameters
    use_lr_scheduler: bool = False  # Enable/disable scheduler usage
    scheduler_type: str = 'plateau'  # 'plateau', 'exponential', 'cosine', 'step'
    
    # Per-optimizer scheduler types (more granular control)
    actor_scheduler_type: str = 'plateau'  # Actor often benefits from adaptive scheduling
    critic_scheduler_type: str = 'exponential'  # Critics usually more stable, can use gradual decay
    alpha_scheduler_type: str = 'plateau'  # Alpha rarely needs scheduling with auto-tuning
    
    # ReduceLROnPlateau scheduler (like DQN v5)
    scheduler_mode: str = 'max'  # 'max' for validation return, 'min' for loss
    scheduler_factor: float = 0.5  # Factor to reduce LR by
    scheduler_patience: int = 10  # Episodes to wait before reducing LR
    scheduler_min_lr: float = 1e-6  # Minimum learning rate
    scheduler_threshold: float = 0.01  # Threshold for measuring improvement
    
    # Per-optimizer plateau scheduler parameters
    actor_scheduler_patience: int = 8  # Actor: more aggressive (shorter patience)
    critic_scheduler_patience: int = 15  # Critics: more conservative (longer patience)
    alpha_scheduler_patience: int = 20  # Alpha: very conservative (longest patience)
    
    actor_scheduler_factor: float = 0.5  # Actor: normal reduction
    critic_scheduler_factor: float = 0.7  # Critics: gentler reduction
    alpha_scheduler_factor: float = 0.8  # Alpha: very gentle reduction
    
    # Exponential scheduler
    scheduler_gamma: float = 0.995  # Decay factor for exponential LR
    
    # Per-optimizer exponential parameters
    actor_scheduler_gamma: float = 0.99  # Actor: faster decay
    critic_scheduler_gamma: float = 0.995  # Critics: moderate decay
    alpha_scheduler_gamma: float = 0.998  # Alpha: slower decay
    
    # Step scheduler
    scheduler_step_size: int = 50  # Episodes between LR reductions
    
    # Cosine annealing scheduler
    scheduler_T_max: int = 100  # Maximum number of iterations
    scheduler_eta_min: float = 1e-6  # Minimum learning rate
    
    # Which optimizers to apply scheduling to
    schedule_actor: bool = True  # Apply scheduler to actor
    schedule_critics: bool = True  # Apply scheduler to critics
    schedule_alpha: bool = False  # Usually not needed with auto-tuning
    
    # Environment-specific parameters
    environment_type: EnvironmentType = EnvironmentType.BASIC
    network_type: NetworkType = NetworkType.SIMPLIFIED
    
    # Lot-based trading parameters (for LOT_BASED environment)
    max_lots: int = 100  # Maximum number of lots to track
    lot_method: str = "FIFO"  # FIFO or LIFO for sell order
    
    # CONTINUOUS ACTION MASKING PARAMETERS
    use_action_guidance: bool = True  # Enable continuous action masking
    action_guidance_strength: float = 0.5  # Strength of action guidance (0-1)
    soft_invalid_penalty: float = 0.01  # Reduced penalty for invalid actions with guidance
    min_action_threshold: float = 0.05  # Minimum action magnitude for meaningful trades
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set target entropy automatically for 1D action space
        # Less conservative than -1.0 to encourage more active trading
        if self.target_entropy == -1.0:
            self.target_entropy = -0.5  # For 1D action space, more active than standard -1.0
    
    def get_environment_config_dict(self) -> dict:
        """Get configuration dictionary for environment creation"""
        return {
            'initial_balance': self.initial_balance,
            'transaction_fee_percent': self.transaction_fee_percent,
            'window_size': self.window_size,
            'num_stock_features': self.num_stock_features,
            'num_portfolio_features': self.num_portfolio_features,
            'num_features': self.num_features,
            'max_position_size': self.max_position_size,
            'minutes_per_day': self.minutes_per_day,
            'use_portfolio_normalization': self.use_portfolio_normalization,
            'portfolio_warmup_episodes': self.portfolio_warmup_episodes,
            'portfolio_update_frequency': self.portfolio_update_frequency,
            'min_trade_amount': self.min_trade_amount,
            'portfolio_scaling': self.portfolio_scaling,
            'invalid_penalty': self.invalid_penalty,
            'max_lots': self.max_lots,
            'lot_method': self.lot_method,
            'use_action_guidance': self.use_action_guidance,
            'action_guidance_strength': self.action_guidance_strength,
            'soft_invalid_penalty': self.soft_invalid_penalty,
            'min_action_threshold': self.min_action_threshold
        }
    
    def get_agent_config_dict(self) -> dict:
        """Get configuration dictionary for agent creation"""
        return {
            'actor_hidden_size': self.actor_hidden_size,
            'critic_hidden_size': self.critic_hidden_size,
            'actor_learning_rate': self.actor_learning_rate,
            'critic_learning_rate': self.critic_learning_rate,
            'alpha_learning_rate': self.alpha_learning_rate,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'update_frequency': self.update_frequency,
            'buffer_size': self.buffer_size,
            'target_entropy': self.target_entropy,
            'alpha_auto_tune': self.alpha_auto_tune,
            'initial_alpha': self.initial_alpha,
            'per_alpha': self.per_alpha,
            'per_beta_start': self.per_beta_start,
            'per_beta_end': self.per_beta_end,
            'per_epsilon': self.per_epsilon,
            'per_beta_annealing_steps': self.per_beta_annealing_steps,
            
            # Learning rate scheduler parameters
            'use_lr_scheduler': self.use_lr_scheduler,
            'scheduler_type': self.scheduler_type,
            'scheduler_mode': self.scheduler_mode,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_min_lr': self.scheduler_min_lr,
            'scheduler_threshold': self.scheduler_threshold,
            'scheduler_gamma': self.scheduler_gamma,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_T_max': self.scheduler_T_max,
            'scheduler_eta_min': self.scheduler_eta_min,
            'schedule_actor': self.schedule_actor,
            'schedule_critics': self.schedule_critics,
            'schedule_alpha': self.schedule_alpha,
            
            # Per-optimizer scheduler parameters
            'actor_scheduler_type': self.actor_scheduler_type,
            'critic_scheduler_type': self.critic_scheduler_type,
            'alpha_scheduler_type': self.alpha_scheduler_type,
            'actor_scheduler_patience': self.actor_scheduler_patience,
            'critic_scheduler_patience': self.critic_scheduler_patience,
            'alpha_scheduler_patience': self.alpha_scheduler_patience,
            'actor_scheduler_factor': self.actor_scheduler_factor,
            'critic_scheduler_factor': self.critic_scheduler_factor,
            'alpha_scheduler_factor': self.alpha_scheduler_factor,
            'actor_scheduler_gamma': self.actor_scheduler_gamma,
            'critic_scheduler_gamma': self.critic_scheduler_gamma,
            'alpha_scheduler_gamma': self.alpha_scheduler_gamma
        }


# Convenience function to create configs with specific environment types
def create_basic_sac_config(**kwargs) -> SACConfig:
    """Create SAC config for basic trading environment with continuous action masking"""
    config = SACConfig()
    config.environment_type = EnvironmentType.BASIC
    # Ensure action masking is enabled for optimal performance
    config.use_action_guidance = True
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_weighted_average_sac_config(**kwargs) -> SACConfig:
    """Create SAC config for weighted average trading environment with continuous action masking"""
    config = SACConfig()
    config.environment_type = EnvironmentType.WEIGHTED_AVERAGE
    # Ensure action masking is enabled for optimal performance
    config.use_action_guidance = True
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_lot_based_sac_config(**kwargs) -> SACConfig:
    """Create SAC config for lot-based trading environment with continuous action masking"""
    config = SACConfig()
    config.environment_type = EnvironmentType.LOT_BASED
    # Ensure action masking is enabled for optimal performance
    config.use_action_guidance = True
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_action_masking_demo_config(environment_type: EnvironmentType = EnvironmentType.WEIGHTED_AVERAGE) -> SACConfig:
    """Create SAC config optimized for demonstrating continuous action masking capabilities"""
    config = SACConfig()
    config.environment_type = environment_type
    
    # Enhanced action masking settings for maximum effectiveness
    config.use_action_guidance = True
    config.action_guidance_strength = 0.7  # Stronger guidance
    config.soft_invalid_penalty = 0.005  # Very soft penalties
    config.min_action_threshold = 0.03  # Lower threshold for more sensitive actions
    
    # Balanced reward scaling for active trading and stable learning  
    config.portfolio_scaling = 0.1  # Balanced scaling - encourages trading while stable for neural networks
    config.invalid_penalty = 0.05  # Consistent with soft penalties
    
    # Optimized training parameters for faster learning with action masking
    config.batch_size = 64  # Smaller batches for more frequent updates
    config.update_frequency = 1  # Update every step
    config.actor_learning_rate = 5e-4  # Slightly higher LR for faster adaptation
    config.target_entropy = -0.3  # More aggressive trading
    
    return config 