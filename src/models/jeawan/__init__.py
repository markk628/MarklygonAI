"""
Jeawan's Trading Models Module
=============================

This module contains advanced reinforcement learning models for stock trading,
including the SAC (Soft Actor-Critic) implementation.

Available Models:
- SAC: Soft Actor-Critic with continuous action space for precise trading
"""

# Try to import SAC module
try:
    from . import sac
    __all__ = ['sac']
except ImportError:
    __all__ = []

__version__ = "1.0.0"
__author__ = "Jeawan"
