"""Adaptive Traffic Signal Control via Hierarchical Multi-Agent RL.

This package implements a hierarchical multi-agent reinforcement learning system
for optimizing city-wide traffic signal control. The system uses low-level agents
to manage individual intersections while high-level agents coordinate traffic flow
across districts.

Key Features:
- Hierarchical RL with two-level agent structure
- Multi-agent cooperation under partial observability
- Transfer learning across different city grid topologies
- Support for SUMO traffic simulation
- Comprehensive evaluation metrics

Authors: Traffic Control Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Traffic Control Research Team"
__email__ = "research@example.com"

# Lazy imports: SUMO-dependent modules may not be available
try:
    from .models.model import (
        HierarchicalTrafficAgent,
        IntersectionAgent,
        DistrictAgent,
    )
except ImportError:
    HierarchicalTrafficAgent = None
    IntersectionAgent = None
    DistrictAgent = None

try:
    from .training.trainer import HierarchicalTrainer
except ImportError:
    HierarchicalTrainer = None

try:
    from .evaluation.metrics import TrafficMetrics
except ImportError:
    TrafficMetrics = None

__all__ = [
    "HierarchicalTrafficAgent",
    "IntersectionAgent",
    "DistrictAgent",
    "HierarchicalTrainer",
    "TrafficMetrics",
]