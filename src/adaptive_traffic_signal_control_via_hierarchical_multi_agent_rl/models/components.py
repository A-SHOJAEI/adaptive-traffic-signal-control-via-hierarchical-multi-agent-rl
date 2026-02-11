"""Custom neural network components and loss functions for hierarchical MARL."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCoordinationLoss(nn.Module):
    """Custom loss function for hierarchical multi-agent coordination.

    This loss combines individual agent performance with coordination quality,
    encouraging agents to both maximize local rewards and coordinate effectively
    with neighboring agents. It uses a weighted combination of:
    1. Standard policy gradient loss (PPO-style)
    2. Value function loss
    3. Coordination penalty based on message consistency
    4. Entropy regularization for exploration

    This is used by the trainer during hierarchical training phases.
    """

    def __init__(
        self,
        policy_coeff: float = 1.0,
        value_coeff: float = 0.5,
        coordination_coeff: float = 0.3,
        entropy_coeff: float = 0.01,
        clip_range: float = 0.2
    ):
        """Initialize hierarchical coordination loss.

        Args:
            policy_coeff: Weight for policy loss component.
            value_coeff: Weight for value function loss.
            coordination_coeff: Weight for coordination penalty.
            entropy_coeff: Weight for entropy regularization.
            clip_range: PPO clipping range.
        """
        super().__init__()
        self.policy_coeff = policy_coeff
        self.value_coeff = value_coeff
        self.coordination_coeff = coordination_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_range = clip_range

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
        communication_messages: Optional[torch.Tensor] = None,
        neighbor_messages: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute hierarchical coordination loss.

        Args:
            log_probs: Log probabilities of selected actions (current policy).
            old_log_probs: Log probabilities from behavior policy.
            advantages: Advantage estimates.
            values: Value function predictions.
            returns: Discounted returns.
            entropies: Action distribution entropies.
            communication_messages: Agent's communication messages.
            neighbor_messages: Neighboring agents' messages.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # PPO policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Entropy loss (for exploration)
        entropy_loss = -entropies.mean()

        # Coordination penalty: encourage message consistency with neighbors
        coordination_loss = torch.tensor(0.0, device=log_probs.device)
        if communication_messages is not None and neighbor_messages is not None:
            # Measure divergence between agent's messages and neighbor averages
            message_diff = communication_messages - neighbor_messages
            coordination_loss = torch.mean(message_diff ** 2)

        # Total loss
        total_loss = (
            self.policy_coeff * policy_loss +
            self.value_coeff * value_loss +
            self.entropy_coeff * entropy_loss +
            self.coordination_coeff * coordination_loss
        )

        # Loss components for logging
        loss_components = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "coordination_loss": coordination_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, loss_components


class SpatialAttentionModule(nn.Module):
    """Spatial attention module for processing traffic grid observations.

    This component learns to focus on critical intersections or areas
    with high congestion, improving the model's ability to prioritize
    important spatial features in the traffic network.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 8
    ):
        """Initialize spatial attention module.

        Args:
            in_channels: Number of input channels.
            reduction_ratio: Channel reduction ratio for efficiency.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(in_channels // reduction_ratio, 1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Attention-weighted tensor.
        """
        batch_size, channels, height, width = x.shape

        # Global pooling
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)

        # Concatenate and compute attention weights
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention_weights = self.fc(combined).view(batch_size, channels, 1, 1)

        # Apply attention
        return x * attention_weights.expand_as(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models.

    Adds positional information to agent embeddings, allowing the model
    to understand the spatial relationships between intersections in the
    traffic network.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        """Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Positionally encoded tensor.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining multiple information sources.

    Used to intelligently combine local observations with communication
    messages from other agents, allowing the model to dynamically balance
    different information sources.
    """

    def __init__(self, input_dim: int):
        """Initialize gated fusion.

        Args:
            input_dim: Dimension of input features.
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh()
        )

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse local and global features.

        Args:
            local_features: Local observation features.
            global_features: Global/communication features.

        Returns:
            Fused features.
        """
        combined = torch.cat([local_features, global_features], dim=-1)
        gate_weights = self.gate(combined)
        transformed = self.transform(combined)
        return gate_weights * local_features + (1 - gate_weights) * transformed


class AdaptiveLearningRateScheduler:
    """Custom learning rate scheduler for hierarchical training.

    Implements a phase-aware learning rate schedule that adapts to the
    three-phase hierarchical training process, with different rates for
    pre-training, coordination learning, and fine-tuning.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        pretrain_lr: float = 3e-4,
        coordination_lr: float = 1e-4,
        finetune_lr: float = 5e-5,
        warmup_steps: int = 1000
    ):
        """Initialize adaptive learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule.
            pretrain_lr: Learning rate for pre-training phase.
            coordination_lr: Learning rate for coordination phase.
            finetune_lr: Learning rate for fine-tuning phase.
            warmup_steps: Number of warmup steps at start of each phase.
        """
        self.optimizer = optimizer
        self.pretrain_lr = pretrain_lr
        self.coordination_lr = coordination_lr
        self.finetune_lr = finetune_lr
        self.warmup_steps = warmup_steps
        self.current_phase = "pretrain"
        self.phase_step = 0

    def set_phase(self, phase: str):
        """Set the current training phase.

        Args:
            phase: One of 'pretrain', 'coordination', or 'finetune'.
        """
        if phase != self.current_phase:
            self.current_phase = phase
            self.phase_step = 0

    def step(self):
        """Step the learning rate scheduler."""
        self.phase_step += 1

        # Select base learning rate based on phase
        if self.current_phase == "pretrain":
            base_lr = self.pretrain_lr
        elif self.current_phase == "coordination":
            base_lr = self.coordination_lr
        else:  # finetune
            base_lr = self.finetune_lr

        # Apply warmup
        if self.phase_step < self.warmup_steps:
            lr = base_lr * (self.phase_step / self.warmup_steps)
        else:
            lr = base_lr

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
