"""Training modules for hierarchical multi-agent RL."""

try:
    from .trainer import HierarchicalTrainer, TrafficEnvironment
except ImportError:
    HierarchicalTrainer = None
    TrafficEnvironment = None

# Synthetic environment is always available (no SUMO dependency)
from .synthetic_env import SyntheticTrafficEnvironment

__all__ = ["HierarchicalTrainer", "TrafficEnvironment", "SyntheticTrafficEnvironment"]