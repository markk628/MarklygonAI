"""
SAC (Soft Actor-Critic) Trading Agent Module
==========================================

A comprehensive SAC implementation for continuous action stock trading.

Key Components:
- SAC: Main Soft Actor-Critic agent
- SACConfig: Configuration class for SAC parameters  
- SACTradingEnvironment: Trading environment for continuous actions
- Actor/Critic: Neural networks for actor-critic architecture
- ReplayBufferGPU: GPU-based experience replay buffer
- train_sac: Main training function
- sac_optimizer: Hyperparameter optimization utilities

Features:
- Continuous action space for precise buy/sell amounts
- Enhanced financial architectures adapted from DQN v5
- GPU-based replay buffer for efficiency
- Portfolio state normalization with warmup phase
- Automatic entropy tuning (SAC-AET)
- Twin critics for stability
- Comprehensive hyperparameter optimization

Example usage:
    ```python
    from src.models.jeawan.sac import train_sac, SACConfig
    import pandas as pd
    
    # Train SAC agent
    results = train_sac(
        data_path="data/TSLA_1min_features.csv",
        cutoff=pd.Timestamp("2023-01-01"),
        num_episodes=500,
        use_preprocessing=True
    )
    
    # Access trained agent
    agent = results['agent']
    ```

Hyperparameter optimization:
    ```python
    from src.models.jeawan.sac.sac_optimizer import run_sac_optimization
    
    best_params = run_sac_optimization(
        data_path="data/TSLA_1min_features.csv",
        cutoff=pd.Timestamp("2023-01-01"),
        n_trials=40
    )
    ```
"""

# Core SAC components from modular architecture
from src.models.jeawan.sac.sac_agent import SAC
from src.models.jeawan.sac.sac_networks import (
    Actor,
    Critic,
    SimplifiedActor,
    SimplifiedCritic,
    PrioritizedReplayBufferGPU,
    create_networks,
    count_parameters
)
from src.models.jeawan.sac.sac_config import (
    SACConfig,
    NetworkType,
    EnvironmentType,
    TradingMode,
    create_basic_sac_config,
    create_weighted_average_sac_config,
    create_lot_based_sac_config,
    create_action_masking_demo_config
)

# Training function and data utilities
from src.models.jeawan.sac.sac import (
    train_sac,
    load_stock_data,
    filter_to_regular_hours
)

# Action masking environments
from src.models.jeawan.sac.sac_environments import (
    create_environment,
    BasicTradingEnvironment,
    WeightedAverageTradingEnvironment, 
    LotBasedTradingEnvironment,
    compare_environments
)

try:
    from src.models.jeawan.sac.sac_optimizer import (
        run_sac_optimization,
        test_sac_parameters,
        save_sac_params_to_file
    )
except ImportError:
    # Optimizer might not be available if optuna is not installed
    pass

try:
    from src.models.jeawan.sac.trainer import main as sac_trainer_main
    from src.models.jeawan.sac.visualization import (
        plot_sac_training_results,
        plot_sac_backtest_results,
        plot_sac_multi_day_comparison
    )
except ImportError:
    # Trainer and visualization might not be available if dependencies are missing
    pass

__all__ = [
    # Core SAC components
    'SAC',
    'SACConfig',
    'NetworkType',
    'EnvironmentType',
    'TradingMode',
    
    # Network architectures
    'Actor',
    'Critic',
    'SimplifiedActor',
    'SimplifiedCritic',
    'PrioritizedReplayBufferGPU',
    'create_networks',
    'count_parameters',
    
    # Configuration factory functions
    'create_basic_sac_config',
    'create_weighted_average_sac_config',
    'create_lot_based_sac_config',
    'create_action_masking_demo_config',
    
    # Action masking environments
    'create_environment',
    'BasicTradingEnvironment',
    'WeightedAverageTradingEnvironment',
    'LotBasedTradingEnvironment',
    'compare_environments',
    
    # Training and data utilities
    'train_sac',
    'load_stock_data',
    'filter_to_regular_hours',
    
    # Optimization utilities (if available)
    'run_sac_optimization',
    'test_sac_parameters', 
    'save_sac_params_to_file',
    
    # Trainer and visualization (if available)
    'sac_trainer_main',
    'plot_sac_training_results',
    'plot_sac_backtest_results',
    'plot_sac_multi_day_comparison'
]

# Version info
__version__ = "1.0.0"
__author__ = "Jeawan (based on DQN v5 implementation by Mark)"
__description__ = "SAC (Soft Actor-Critic) trading agent for continuous action stock trading" 