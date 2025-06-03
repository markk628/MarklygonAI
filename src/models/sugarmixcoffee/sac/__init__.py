"""
Soft Actor-Critic (SAC) implementation for stock trading.

This module implements both discrete and continuous SAC agents for minute-level stock data.
It uses prioritized experience replay and the same trading environment infrastructure
but offers two different action spaces:

1. Discrete SAC: Hold (0), Buy (1), Sell (2) - binary trading decisions
2. Continuous SAC: [-1, 1] - granular position sizing decisions

Key Features:
- Actor-Critic architecture with twin Q-networks
- Entropy regularization for better exploration
- Prioritized Experience Replay
- AdamW optimizer with learning rate scheduling
- Same preprocessing pipeline as DQN
- Database integration for result tracking

Main Components:
- SACAgent: Discrete action SAC agent
- ContinuousSACAgent: Continuous action SAC agent with position sizing
- ActorNetwork/ContinuousActorNetwork: Policy networks
- CriticNetwork/ContinuousCriticNetwork: Q-value estimation networks
- SACConfig/ContinuousSACConfig: Configuration dataclasses
- train_sac/train_continuous_sac: Main training functions
"""

# Discrete SAC (original implementation)
from .sac import (
    SACAgent,
    SACConfig,
    ActorNetwork,
    CriticNetwork,
    train_sac
)

# Continuous SAC (position sizing implementation)
from .continuous_sac import (
    ContinuousSACConfig,
    ContinuousActorNetwork,
    ContinuousCriticNetwork,
    ContinuousPrioritizedReplayBuffer,
    ContinuousTradingEnvironment
)

from .continuous_sac_agent import (
    ContinuousSACAgent,
    train_continuous_sac
)

# Visualization and training utilities
from .trainer import (
    plot_sac_training_results,
    plot_sac_backtest_results,
    compare_strategies
)

__all__ = [
    # Discrete SAC
    'SACAgent',
    'SACConfig', 
    'ActorNetwork',
    'CriticNetwork',
    'train_sac',
    
    # Continuous SAC
    'ContinuousSACAgent',
    'ContinuousSACConfig',
    'ContinuousActorNetwork',
    'ContinuousCriticNetwork',
    'ContinuousPrioritizedReplayBuffer',
    'ContinuousTradingEnvironment',
    'train_continuous_sac',
    
    # Utilities
    'plot_sac_training_results',
    'plot_sac_backtest_results',
    'compare_strategies'
]

__version__ = '2.0.0'  # Updated for continuous action support
__author__ = 'MarklygonAI' 