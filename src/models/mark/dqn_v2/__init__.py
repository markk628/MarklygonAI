"""
Double Dueling DQN with Prioritized Experience Replay for Stock Trading
"""

from .config import TradingConfig
from .networks import DuelingNetwork
from .dqn_v7 import (
    EnhancedTradingConfig,
    PrioritizedReplayBufferGPU,
    EnhancedTradingEnvironment,
    EnhancedDoubleDuelingDQN,
    train_enhanced_dqn
)

# Aliases for backward compatibility
TradingEnvironment = EnhancedTradingEnvironment
DoubleDuelingDQN = EnhancedDoubleDuelingDQN
train_dqn = train_enhanced_dqn

__all__ = [
    'TradingConfig',
    'EnhancedTradingConfig',
    'DuelingNetwork', 
    'PrioritizedReplayBufferGPU',
    'TradingEnvironment',
    'EnhancedTradingEnvironment',
    'DoubleDuelingDQN',
    'EnhancedDoubleDuelingDQN',
    'train_dqn',
    'train_enhanced_dqn'
] 