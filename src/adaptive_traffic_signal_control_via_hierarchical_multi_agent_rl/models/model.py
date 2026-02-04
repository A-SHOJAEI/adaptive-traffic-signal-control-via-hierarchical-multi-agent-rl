"""Hierarchical multi-agent RL models for traffic signal control."""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..utils.config import Config

logger = logging.getLogger(__name__)


class AttentionNetwork(nn.Module):
    """Multi-head attention network for processing sequential data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        """Initialize attention network.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # Project input to hidden dimension
        x = self.input_projection(x)

        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class CommunicationNetwork(nn.Module):
    """Communication network for inter-agent message passing."""

    def __init__(
        self,
        agent_embedding_dim: int,
        communication_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2
    ) -> None:
        """Initialize communication network.

        Args:
            agent_embedding_dim: Dimension of agent embeddings.
            communication_dim: Dimension of communication messages.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of network layers.
        """
        super().__init__()
        self.agent_embedding_dim = agent_embedding_dim
        self.communication_dim = communication_dim

        # Message encoding network
        self.message_encoder = self._build_mlp(
            agent_embedding_dim,
            communication_dim,
            hidden_dim,
            num_layers
        )

        # Message aggregation network
        self.message_aggregator = self._build_mlp(
            communication_dim,
            communication_dim,
            hidden_dim,
            num_layers
        )

        # Message decoding network
        self.message_decoder = self._build_mlp(
            communication_dim + agent_embedding_dim,
            agent_embedding_dim,
            hidden_dim,
            num_layers
        )

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int
    ) -> nn.Module:
        """Build multi-layer perceptron.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of layers.

        Returns:
            MLP module.
        """
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(
        self,
        agent_embeddings: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for communication.

        Args:
            agent_embeddings: Agent embedding tensor (batch_size, num_agents, embedding_dim).
            adjacency_matrix: Adjacency matrix (num_agents, num_agents).

        Returns:
            Tuple of (updated_embeddings, communication_messages).
        """
        batch_size, num_agents, embedding_dim = agent_embeddings.shape

        # Encode messages from each agent
        messages = self.message_encoder(agent_embeddings)  # (batch_size, num_agents, comm_dim)

        # Aggregate messages based on adjacency matrix
        adjacency_matrix = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        aggregated_messages = torch.bmm(adjacency_matrix.float(), messages)

        # Normalize by number of neighbors
        neighbor_counts = adjacency_matrix.sum(dim=-1, keepdim=True)
        neighbor_counts = torch.clamp(neighbor_counts, min=1.0)
        aggregated_messages = aggregated_messages / neighbor_counts

        # Process aggregated messages
        processed_messages = self.message_aggregator(aggregated_messages)

        # Combine with agent embeddings and decode
        combined_input = torch.cat([processed_messages, agent_embeddings], dim=-1)
        updated_embeddings = self.message_decoder(combined_input)

        return updated_embeddings, processed_messages


class IntersectionAgent(nn.Module):
    """Low-level agent for individual intersection control."""

    def __init__(self, config: Config) -> None:
        """Initialize intersection agent.

        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.observation_dim = config.agents.low_level.observation_space_dim
        self.action_dim = config.agents.low_level.action_space_size
        self.hidden_layers = config.model.low_level_net.hidden_layers

        # Feature extraction network
        self.feature_extractor = self._build_feature_extractor()

        # LSTM for temporal processing
        if config.model.low_level_net.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.hidden_layers[-1],
                hidden_size=config.model.low_level_net.lstm_hidden_size,
                batch_first=True
            )
            self.lstm_hidden_size = config.model.low_level_net.lstm_hidden_size
        else:
            self.lstm = None
            self.lstm_hidden_size = self.hidden_layers[-1]

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[-1], self.action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[-1], 1)
        )

        # Communication embedding
        communication_dim = config.training.hierarchical.communication_dim
        self.communication_encoder = nn.Linear(self.lstm_hidden_size, communication_dim)
        self.communication_decoder = nn.Linear(communication_dim, self.lstm_hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction network.

        Returns:
            Feature extraction module.
        """
        layers = []
        input_dim = self.observation_dim
        activation = getattr(nn, self.config.model.low_level_net.activation.capitalize())()

        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights.

        Args:
            module: Neural network module.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    def forward(
        self,
        observations: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        communication_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Args:
            observations: Observation tensor.
            hidden_state: LSTM hidden state.
            communication_input: Communication input from higher level.

        Returns:
            Tuple of (action_logits, values, communication_output, new_hidden_state).
        """
        batch_size = observations.shape[0]

        # Extract features
        features = self.feature_extractor(observations)

        # Process through LSTM if available
        if self.lstm is not None:
            if features.dim() == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
            features, hidden_state = self.lstm(features, hidden_state)
            features = features.squeeze(1)  # Remove sequence dimension
        else:
            hidden_state = None

        # Apply communication input if provided
        if communication_input is not None:
            decoded_communication = self.communication_decoder(communication_input)
            features = features + decoded_communication

        # Compute actor and critic outputs
        action_logits = self.actor(features)
        values = self.critic(features)

        # Generate communication output
        communication_output = self.communication_encoder(features)

        return action_logits, values, communication_output, hidden_state

    def get_action_distribution(self, action_logits: torch.Tensor) -> torch.distributions.Distribution:
        """Get action distribution from logits.

        Args:
            action_logits: Action logits tensor.

        Returns:
            Categorical distribution over actions.
        """
        return torch.distributions.Categorical(probs=action_logits)


class DistrictAgent(nn.Module):
    """High-level agent for district-wide coordination."""

    def __init__(self, config: Config) -> None:
        """Initialize district agent.

        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.observation_dim = config.agents.high_level.observation_space_dim
        self.action_dim = config.agents.high_level.action_space_size
        self.hidden_layers = config.model.high_level_net.hidden_layers
        self.communication_dim = config.training.hierarchical.communication_dim

        # Attention network for processing multiple intersection states
        if config.model.high_level_net.use_transformer:
            self.attention = AttentionNetwork(
                input_dim=self.communication_dim,
                hidden_dim=self.hidden_layers[0],
                num_heads=config.model.high_level_net.attention_heads
            )
        else:
            self.attention = None

        # Feature aggregation network
        self.feature_aggregator = self._build_feature_aggregator()

        # Actor network (continuous policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(self.hidden_layers[-1], self.hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[-1], self.action_dim)
        )

        self.actor_logstd = nn.Parameter(torch.zeros(self.action_dim))

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_layers[-1], self.hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[-1], 1)
        )

        # Communication network for coordinating with low-level agents
        district_size = config.agents.high_level.district_size
        num_intersections = district_size[0] * district_size[1]
        self.communication_network = CommunicationNetwork(
            agent_embedding_dim=self.communication_dim,
            communication_dim=self.communication_dim,
            hidden_dim=self.hidden_layers[0]
        )

        # Coordination output network
        self.coordination_output = nn.Sequential(
            nn.Linear(self.hidden_layers[-1], self.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0], num_intersections * self.communication_dim)
        )

        self.num_intersections = num_intersections

        # Initialize weights
        self.apply(self._init_weights)

    def _build_feature_aggregator(self) -> nn.Module:
        """Build feature aggregation network.

        Returns:
            Feature aggregation module.
        """
        layers = []
        input_dim = self.observation_dim
        activation = getattr(nn, self.config.model.high_level_net.activation.capitalize())()

        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights.

        Args:
            module: Neural network module.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        observations: torch.Tensor,
        intersection_communications: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observations: District-level observations.
            intersection_communications: Communications from intersection agents.
            adjacency_matrix: Adjacency matrix for intersections.

        Returns:
            Tuple of (action_mean, action_logstd, values, coordination_signals).
        """
        batch_size = observations.shape[0]

        # Process intersection communications through attention if available
        if self.attention is not None:
            attended_communications = self.attention(intersection_communications)
            # Global pooling to get district-level representation
            district_features = torch.mean(attended_communications, dim=1)
        else:
            # Simple mean pooling
            district_features = torch.mean(intersection_communications, dim=1)

        # Combine with district observations
        combined_features = torch.cat([observations, district_features], dim=-1)

        # Handle dimension mismatch
        if combined_features.shape[-1] != self.observation_dim:
            # Project to correct dimension
            projection = nn.Linear(combined_features.shape[-1], self.observation_dim).to(combined_features.device)
            combined_features = projection(combined_features)

        # Aggregate features
        aggregated_features = self.feature_aggregator(combined_features)

        # Compute actor outputs
        action_mean = self.actor_mean(aggregated_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Compute critic output
        values = self.critic(aggregated_features)

        # Generate coordination signals for low-level agents
        coordination_flat = self.coordination_output(aggregated_features)
        coordination_signals = coordination_flat.view(
            batch_size, self.num_intersections, self.communication_dim
        )

        # Process coordination through communication network
        updated_communications, processed_messages = self.communication_network(
            intersection_communications, adjacency_matrix
        )

        return action_mean, action_logstd, values, updated_communications

    def get_action_distribution(
        self,
        action_mean: torch.Tensor,
        action_logstd: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Get action distribution.

        Args:
            action_mean: Action mean tensor.
            action_logstd: Action log standard deviation tensor.

        Returns:
            Normal distribution over continuous actions.
        """
        action_std = torch.exp(action_logstd)
        return torch.distributions.Normal(action_mean, action_std)


class HierarchicalTrafficAgent(nn.Module):
    """Hierarchical traffic control agent combining intersection and district agents."""

    def __init__(self, config: Config) -> None:
        """Initialize hierarchical traffic agent.

        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.hierarchy_levels = config.agents.hierarchy_levels
        self.coordination_frequency = config.training.hierarchical.coordination_frequency

        # Create intersection agents
        self.intersection_agents = nn.ModuleDict()
        self.intersection_hidden_states = {}

        # Create district agents
        self.district_agents = nn.ModuleDict()

        # Communication network between levels
        communication_dim = config.training.hierarchical.communication_dim
        self.inter_level_communication = CommunicationNetwork(
            agent_embedding_dim=communication_dim,
            communication_dim=communication_dim,
            hidden_dim=config.model.communication_net.hidden_layers[0]
        )

        # Step counter for coordination frequency
        self.step_count = 0

        logger.info(f"Initialized hierarchical agent with {self.hierarchy_levels} levels")

    def add_intersection_agent(self, agent_id: str) -> None:
        """Add intersection agent to the hierarchy.

        Args:
            agent_id: Unique identifier for the intersection agent.
        """
        self.intersection_agents[agent_id] = IntersectionAgent(self.config)
        self.intersection_hidden_states[agent_id] = None
        logger.debug(f"Added intersection agent: {agent_id}")

    def add_district_agent(self, agent_id: str) -> None:
        """Add district agent to the hierarchy.

        Args:
            agent_id: Unique identifier for the district agent.
        """
        self.district_agents[agent_id] = DistrictAgent(self.config)
        logger.debug(f"Added district agent: {agent_id}")

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        adjacency_matrices: Dict[str, torch.Tensor],
        agent_mappings: Dict[str, List[str]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through hierarchical architecture.

        Args:
            observations: Dictionary of observations for each agent.
            adjacency_matrices: Adjacency matrices for communication.
            agent_mappings: Mapping from district agents to intersection agents.

        Returns:
            Dictionary of (actions, values) for each agent.
        """
        self.step_count += 1
        should_coordinate = (self.step_count % self.coordination_frequency) == 0

        outputs = {}
        communication_messages = {}

        # First pass: Forward through intersection agents
        for agent_id, agent in self.intersection_agents.items():
            obs = observations[agent_id]
            hidden_state = self.intersection_hidden_states.get(agent_id)

            action_logits, values, comm_output, new_hidden_state = agent(
                obs, hidden_state
            )

            outputs[agent_id] = (action_logits, values)
            communication_messages[agent_id] = comm_output
            self.intersection_hidden_states[agent_id] = new_hidden_state

        # Second pass: Forward through district agents (if coordination step)
        if should_coordinate:
            for district_id, district_agent in self.district_agents.items():
                # Get observations and communications from managed intersections
                managed_intersections = agent_mappings.get(district_id, [])

                if not managed_intersections:
                    continue

                # Collect communications from managed intersections
                intersection_comms = []
                for intersection_id in managed_intersections:
                    if intersection_id in communication_messages:
                        intersection_comms.append(communication_messages[intersection_id])

                if intersection_comms:
                    intersection_communications = torch.stack(intersection_comms, dim=1)
                    district_obs = observations[district_id]
                    adjacency_matrix = adjacency_matrices.get(district_id,
                                                            torch.eye(len(managed_intersections)))

                    action_mean, action_logstd, values, coordination_signals = district_agent(
                        district_obs, intersection_communications, adjacency_matrix
                    )

                    outputs[district_id] = (action_mean, action_logstd, values)

                    # Send coordination signals back to intersection agents
                    for i, intersection_id in enumerate(managed_intersections):
                        if intersection_id in self.intersection_agents:
                            # Re-forward intersection agent with coordination signal
                            coord_signal = coordination_signals[:, i:i+1, :]
                            obs = observations[intersection_id]
                            hidden_state = self.intersection_hidden_states.get(intersection_id)

                            action_logits, values, _, new_hidden_state = self.intersection_agents[intersection_id](
                                obs, hidden_state, coord_signal.squeeze(1)
                            )

                            outputs[intersection_id] = (action_logits, values)
                            self.intersection_hidden_states[intersection_id] = new_hidden_state

        return outputs

    def get_intersection_agents(self) -> Dict[str, IntersectionAgent]:
        """Get intersection agents.

        Returns:
            Dictionary of intersection agents.
        """
        return dict(self.intersection_agents)

    def get_district_agents(self) -> Dict[str, DistrictAgent]:
        """Get district agents.

        Returns:
            Dictionary of district agents.
        """
        return dict(self.district_agents)

    def reset_hidden_states(self) -> None:
        """Reset hidden states for all LSTM-based agents."""
        for agent_id in self.intersection_hidden_states:
            self.intersection_hidden_states[agent_id] = None

        logger.debug("Reset hidden states for all agents")

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "intersection_agents": {aid: agent.state_dict()
                                  for aid, agent in self.intersection_agents.items()},
            "district_agents": {aid: agent.state_dict()
                              for aid, agent in self.district_agents.items()},
            "inter_level_communication": self.inter_level_communication.state_dict(),
            "step_count": self.step_count,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location="cpu")

        # Load intersection agents
        for aid, state_dict in checkpoint["intersection_agents"].items():
            if aid in self.intersection_agents:
                self.intersection_agents[aid].load_state_dict(state_dict)

        # Load district agents
        for aid, state_dict in checkpoint["district_agents"].items():
            if aid in self.district_agents:
                self.district_agents[aid].load_state_dict(state_dict)

        # Load inter-level communication
        self.inter_level_communication.load_state_dict(
            checkpoint["inter_level_communication"]
        )

        self.step_count = checkpoint.get("step_count", 0)

        logger.info(f"Loaded checkpoint from {path}")


class TrafficEnvironmentWrapper:
    """Wrapper for integrating hierarchical agents with training environments."""

    def __init__(
        self,
        config: Config,
        hierarchical_agent: HierarchicalTrafficAgent
    ) -> None:
        """Initialize environment wrapper.

        Args:
            config: Configuration object.
            hierarchical_agent: Hierarchical traffic agent.
        """
        self.config = config
        self.hierarchical_agent = hierarchical_agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move agent to device
        self.hierarchical_agent.to(self.device)

    def predict(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, np.ndarray]:
        """Predict actions for all agents.

        Args:
            observations: Dictionary of observations for each agent.
            deterministic: Whether to use deterministic actions.

        Returns:
            Dictionary of actions for each agent.
        """
        # Convert observations to tensors
        tensor_observations = {}
        adjacency_matrices = {}

        for agent_id, obs in observations.items():
            tensor_observations[agent_id] = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Create agent mappings (simplified - in practice this would be more complex)
        agent_mappings = self._create_agent_mappings()

        # Create adjacency matrices
        for district_id in self.hierarchical_agent.district_agents.keys():
            managed_intersections = agent_mappings.get(district_id, [])
            if managed_intersections:
                adjacency_matrices[district_id] = torch.eye(len(managed_intersections)).to(self.device)

        with torch.no_grad():
            outputs = self.hierarchical_agent(
                tensor_observations, adjacency_matrices, agent_mappings
            )

        # Convert outputs to actions
        actions = {}
        for agent_id, output in outputs.items():
            if agent_id in self.hierarchical_agent.intersection_agents:
                # Discrete actions for intersection agents
                action_logits, _ = output
                if deterministic:
                    action = torch.argmax(action_logits, dim=-1)
                else:
                    dist = torch.distributions.Categorical(probs=action_logits)
                    action = dist.sample()
                actions[agent_id] = action.cpu().numpy()

            elif agent_id in self.hierarchical_agent.district_agents:
                # Continuous actions for district agents
                action_mean, action_logstd, _ = output
                if deterministic:
                    action = action_mean
                else:
                    action_std = torch.exp(action_logstd)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                actions[agent_id] = action.cpu().numpy()

        return actions

    def _create_agent_mappings(self) -> Dict[str, List[str]]:
        """Create mapping from district agents to intersection agents.

        Returns:
            Dictionary mapping district agent IDs to lists of intersection agent IDs.
        """
        # This is a simplified implementation
        # In practice, this would be based on the actual network topology
        mappings = {}
        intersection_ids = list(self.hierarchical_agent.intersection_agents.keys())
        district_ids = list(self.hierarchical_agent.district_agents.keys())

        # Simple grid-based assignment
        district_size = self.config.agents.high_level.district_size
        intersections_per_district = district_size[0] * district_size[1]

        for i, district_id in enumerate(district_ids):
            start_idx = i * intersections_per_district
            end_idx = min(start_idx + intersections_per_district, len(intersection_ids))
            mappings[district_id] = intersection_ids[start_idx:end_idx]

        return mappings