"""
Soft Actor-Critic (SAC) implementation for stock trading.

This module implements a SAC agent that can be trained on minute-level stock data.
It uses the same prioritized experience replay and trading environment as the DQN implementation
but with an actor-critic architecture and entropy regularization.

Key Features:
- Actor-Critic architecture with twin Q-networks
- Entropy regularization for better exploration
- Prioritized Experience Replay
- AdamW optimizer with learning rate scheduling
- Same preprocessing pipeline as DQN
- Database integration for result tracking

Main Components:
- SACAgent: The main agent class
- ActorNetwork: Policy network for action selection
- CriticNetwork: Q-value estimation networks
- SACConfig: Configuration dataclass
- train_sac: Main training function
"""

from .sac import (
    SACAgent,
    SACConfig,
    ActorNetwork,
    CriticNetwork,
    train_sac
)

from .trainer import (
    plot_sac_training_results,
    plot_sac_backtest_results,
    compare_strategies
)

__all__ = [
    'SACAgent',
    'SACConfig', 
    'ActorNetwork',
    'CriticNetwork',
    'train_sac',
    'plot_sac_training_results',
    'plot_sac_backtest_results',
    'compare_strategies'
]

__version__ = '1.0.0'
__author__ = 'MarklygonAI' 