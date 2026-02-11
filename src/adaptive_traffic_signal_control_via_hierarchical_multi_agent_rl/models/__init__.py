"""Model implementations for hierarchical multi-agent traffic control."""

try:
    from .model import (
        HierarchicalTrafficAgent,
        IntersectionAgent,
        DistrictAgent,
        CommunicationNetwork,
        AttentionNetwork,
    )
    from .components import (
        HierarchicalCoordinationLoss,
        SpatialAttentionModule,
        PositionalEncoding,
        GatedFusion,
        AdaptiveLearningRateScheduler,
    )
except ImportError:
    HierarchicalTrafficAgent = None
    IntersectionAgent = None
    DistrictAgent = None
    CommunicationNetwork = None
    AttentionNetwork = None
    HierarchicalCoordinationLoss = None
    SpatialAttentionModule = None
    PositionalEncoding = None
    GatedFusion = None
    AdaptiveLearningRateScheduler = None

__all__ = [
    "HierarchicalTrafficAgent",
    "IntersectionAgent",
    "DistrictAgent",
    "CommunicationNetwork",
    "AttentionNetwork",
    "HierarchicalCoordinationLoss",
    "SpatialAttentionModule",
    "PositionalEncoding",
    "GatedFusion",
    "AdaptiveLearningRateScheduler",
]