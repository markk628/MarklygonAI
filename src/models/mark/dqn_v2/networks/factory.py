import torch.nn as nn
from ..config import ArchitectureType
from .original import DuelingNetworkOriginal
from .improved import ImprovedDuelingNetwork
from .hybrid import HybridCNNLSTMNetwork


def create_network(config) -> nn.Module:
    """Factory function to create network based on config"""
    if config.architecture_type == ArchitectureType.ORIGINAL:
        return DuelingNetworkOriginal(config)
    elif config.architecture_type == ArchitectureType.IMPROVED:
        return ImprovedDuelingNetwork(config)
    elif config.architecture_type == ArchitectureType.HYBRID:
        return HybridCNNLSTMNetwork(config)
    else:
        raise ValueError(f"Unknown architecture type: {config.architecture_type}")


# Alias for backward compatibility
class DuelingNetwork(ImprovedDuelingNetwork):
    """Use improved architecture by default"""
    pass 