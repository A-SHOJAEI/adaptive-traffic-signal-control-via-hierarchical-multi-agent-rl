"""Tests for model implementations."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import (
    AttentionNetwork,
    CommunicationNetwork,
    IntersectionAgent,
    DistrictAgent,
    HierarchicalTrafficAgent,
    TrafficEnvironmentWrapper
)


class TestAttentionNetwork:
    """Test cases for AttentionNetwork."""

    def test_initialization(self):
        """Test AttentionNetwork initialization."""
        network = AttentionNetwork(
            input_dim=64,
            hidden_dim=128,
            num_heads=8,
            dropout=0.1
        )

        assert network.input_dim == 64
        assert network.hidden_dim == 128
        assert network.num_heads == 8
        assert isinstance(network.attention, torch.nn.MultiheadAttention)

    def test_forward_pass(self):
        """Test AttentionNetwork forward pass."""
        network = AttentionNetwork(
            input_dim=32,
            hidden_dim=64,
            num_heads=4
        )

        batch_size = 2
        seq_len = 5
        input_tensor = torch.randn(batch_size, seq_len, 32)

        output = network(input_tensor)

        assert output.shape == (batch_size, seq_len, 64)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_forward_pass_different_shapes(self):
        """Test AttentionNetwork with different input shapes."""
        network = AttentionNetwork(input_dim=16, hidden_dim=32, num_heads=2)

        # Test single sequence
        single_input = torch.randn(1, 3, 16)
        output = network(single_input)
        assert output.shape == (1, 3, 32)

        # Test longer sequence
        long_input = torch.randn(1, 10, 16)
        output = network(long_input)
        assert output.shape == (1, 10, 32)


class TestCommunicationNetwork:
    """Test cases for CommunicationNetwork."""

    def test_initialization(self):
        """Test CommunicationNetwork initialization."""
        network = CommunicationNetwork(
            agent_embedding_dim=64,
            communication_dim=32,
            hidden_dim=128
        )

        assert network.agent_embedding_dim == 64
        assert network.communication_dim == 32
        assert isinstance(network.message_encoder, torch.nn.Sequential)

    def test_build_mlp(self):
        """Test MLP building."""
        network = CommunicationNetwork(
            agent_embedding_dim=64,
            communication_dim=32
        )

        mlp = network._build_mlp(input_dim=10, output_dim=5, hidden_dim=20, num_layers=2)

        assert isinstance(mlp, torch.nn.Sequential)
        assert len(mlp) > 0

        # Test forward pass
        test_input = torch.randn(3, 10)
        output = mlp(test_input)
        assert output.shape == (3, 5)

    def test_forward_pass(self):
        """Test CommunicationNetwork forward pass."""
        network = CommunicationNetwork(
            agent_embedding_dim=32,
            communication_dim=16,
            hidden_dim=64
        )

        batch_size = 2
        num_agents = 4
        agent_embeddings = torch.randn(batch_size, num_agents, 32)
        adjacency_matrix = torch.eye(num_agents)

        updated_embeddings, messages = network(agent_embeddings, adjacency_matrix)

        assert updated_embeddings.shape == (batch_size, num_agents, 32)
        assert messages.shape == (batch_size, num_agents, 16)
        assert not torch.any(torch.isnan(updated_embeddings))
        assert not torch.any(torch.isnan(messages))

    def test_communication_with_sparse_adjacency(self):
        """Test communication with sparse adjacency matrix."""
        network = CommunicationNetwork(
            agent_embedding_dim=16,
            communication_dim=8
        )

        num_agents = 3
        agent_embeddings = torch.randn(1, num_agents, 16)

        # Create sparse adjacency matrix (only agent 0 connected to agent 1)
        adjacency_matrix = torch.zeros(num_agents, num_agents)
        adjacency_matrix[0, 1] = 1.0
        adjacency_matrix[1, 0] = 1.0

        updated_embeddings, messages = network(agent_embeddings, adjacency_matrix)

        assert updated_embeddings.shape == (1, num_agents, 16)
        assert messages.shape == (1, num_agents, 8)


class TestIntersectionAgent:
    """Test cases for IntersectionAgent."""

    def test_initialization(self, test_config):
        """Test IntersectionAgent initialization."""
        agent = IntersectionAgent(test_config)

        assert agent.config == test_config
        assert agent.observation_dim == test_config.agents.low_level.observation_space_dim
        assert agent.action_dim == test_config.agents.low_level.action_space_size
        assert isinstance(agent.feature_extractor, torch.nn.Sequential)
        assert isinstance(agent.actor, torch.nn.Sequential)
        assert isinstance(agent.critic, torch.nn.Sequential)

    def test_forward_pass(self, test_config):
        """Test IntersectionAgent forward pass."""
        agent = IntersectionAgent(test_config)

        batch_size = 3
        obs_dim = test_config.agents.low_level.observation_space_dim
        observations = torch.randn(batch_size, obs_dim)

        action_logits, values, comm_output, hidden_state = agent(observations)

        assert action_logits.shape == (batch_size, test_config.agents.low_level.action_space_size)
        assert values.shape == (batch_size, 1)
        assert comm_output.shape == (batch_size, test_config.training.hierarchical.communication_dim)

        # Check that action probabilities sum to 1
        assert torch.allclose(action_logits.sum(dim=-1), torch.ones(batch_size))

    def test_forward_with_lstm(self, test_config):
        """Test IntersectionAgent forward pass with LSTM."""
        agent = IntersectionAgent(test_config)

        batch_size = 2
        obs_dim = test_config.agents.low_level.observation_space_dim
        observations = torch.randn(batch_size, obs_dim)

        # First forward pass
        action_logits1, values1, comm1, hidden1 = agent(observations)

        # Second forward pass with hidden state
        action_logits2, values2, comm2, hidden2 = agent(observations, hidden1)

        assert action_logits2.shape == action_logits1.shape
        assert values2.shape == values1.shape
        assert hidden2 is not None

    def test_forward_with_communication_input(self, test_config):
        """Test IntersectionAgent with communication input."""
        agent = IntersectionAgent(test_config)

        batch_size = 2
        obs_dim = test_config.agents.low_level.observation_space_dim
        comm_dim = test_config.training.hierarchical.communication_dim

        observations = torch.randn(batch_size, obs_dim)
        communication_input = torch.randn(batch_size, comm_dim)

        action_logits, values, comm_output, hidden_state = agent(
            observations, communication_input=communication_input
        )

        assert action_logits.shape == (batch_size, test_config.agents.low_level.action_space_size)
        assert values.shape == (batch_size, 1)

    def test_get_action_distribution(self, test_config):
        """Test action distribution creation."""
        agent = IntersectionAgent(test_config)

        batch_size = 2
        action_logits = torch.softmax(torch.randn(batch_size, 4), dim=-1)

        distribution = agent.get_action_distribution(action_logits)

        assert isinstance(distribution, torch.distributions.Categorical)
        assert distribution.batch_shape == (batch_size,)

        # Test sampling
        actions = distribution.sample()
        assert actions.shape == (batch_size,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 4)

    def test_weight_initialization(self, test_config):
        """Test proper weight initialization."""
        agent = IntersectionAgent(test_config)

        # Check that weights are not all zeros or ones
        for module in agent.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.all(module.weight == 0)
                assert not torch.all(module.weight == 1)


class TestDistrictAgent:
    """Test cases for DistrictAgent."""

    def test_initialization(self, test_config):
        """Test DistrictAgent initialization."""
        agent = DistrictAgent(test_config)

        assert agent.config == test_config
        assert agent.observation_dim == test_config.agents.high_level.observation_space_dim
        assert agent.action_dim == test_config.agents.high_level.action_space_size
        assert isinstance(agent.feature_aggregator, torch.nn.Sequential)
        assert isinstance(agent.actor_mean, torch.nn.Sequential)
        assert isinstance(agent.critic, torch.nn.Sequential)

    def test_forward_pass(self, test_config):
        """Test DistrictAgent forward pass."""
        agent = DistrictAgent(test_config)

        batch_size = 2
        obs_dim = test_config.agents.high_level.observation_space_dim
        comm_dim = test_config.training.hierarchical.communication_dim

        district_size = test_config.agents.high_level.district_size
        num_intersections = district_size[0] * district_size[1]

        observations = torch.randn(batch_size, obs_dim)
        intersection_comms = torch.randn(batch_size, num_intersections, comm_dim)
        adjacency_matrix = torch.eye(num_intersections)

        action_mean, action_logstd, values, coordination_signals = agent(
            observations, intersection_comms, adjacency_matrix
        )

        assert action_mean.shape == (batch_size, test_config.agents.high_level.action_space_size)
        assert action_logstd.shape == (batch_size, test_config.agents.high_level.action_space_size)
        assert values.shape == (batch_size, 1)
        assert coordination_signals.shape == (batch_size, num_intersections, comm_dim)

    def test_forward_with_attention(self, test_config):
        """Test DistrictAgent forward pass with attention mechanism."""
        # Enable transformer in config
        test_config.model.high_level_net.use_transformer = True

        agent = DistrictAgent(test_config)

        batch_size = 1
        obs_dim = test_config.agents.high_level.observation_space_dim
        comm_dim = test_config.training.hierarchical.communication_dim

        district_size = test_config.agents.high_level.district_size
        num_intersections = district_size[0] * district_size[1]

        observations = torch.randn(batch_size, obs_dim)
        intersection_comms = torch.randn(batch_size, num_intersections, comm_dim)
        adjacency_matrix = torch.eye(num_intersections)

        action_mean, action_logstd, values, coordination_signals = agent(
            observations, intersection_comms, adjacency_matrix
        )

        assert action_mean.shape == (batch_size, test_config.agents.high_level.action_space_size)
        assert coordination_signals.shape == (batch_size, num_intersections, comm_dim)

    def test_get_action_distribution(self, test_config):
        """Test action distribution creation for continuous actions."""
        agent = DistrictAgent(test_config)

        batch_size = 2
        action_dim = test_config.agents.high_level.action_space_size

        action_mean = torch.randn(batch_size, action_dim)
        action_logstd = torch.zeros(batch_size, action_dim)  # std = 1

        distribution = agent.get_action_distribution(action_mean, action_logstd)

        assert isinstance(distribution, torch.distributions.Normal)
        assert distribution.batch_shape == (batch_size,)
        assert distribution.event_shape == (action_dim,)

        # Test sampling
        actions = distribution.sample()
        assert actions.shape == (batch_size, action_dim)

    def test_communication_network_integration(self, test_config):
        """Test communication network integration."""
        agent = DistrictAgent(test_config)

        batch_size = 1
        comm_dim = test_config.training.hierarchical.communication_dim
        num_agents = 4

        agent_embeddings = torch.randn(batch_size, num_agents, comm_dim)
        adjacency_matrix = torch.eye(num_agents)

        updated_comms, processed_messages = agent.communication_network(
            agent_embeddings, adjacency_matrix
        )

        assert updated_comms.shape == (batch_size, num_agents, comm_dim)
        assert processed_messages.shape == (batch_size, num_agents, comm_dim)


class TestHierarchicalTrafficAgent:
    """Test cases for HierarchicalTrafficAgent."""

    def test_initialization(self, test_config):
        """Test HierarchicalTrafficAgent initialization."""
        agent = HierarchicalTrafficAgent(test_config)

        assert agent.config == test_config
        assert agent.hierarchy_levels == test_config.agents.hierarchy_levels
        assert isinstance(agent.intersection_agents, torch.nn.ModuleDict)
        assert isinstance(agent.district_agents, torch.nn.ModuleDict)

    def test_add_intersection_agent(self, test_config):
        """Test adding intersection agent."""
        agent = HierarchicalTrafficAgent(test_config)

        agent_id = "intersection_test_1"
        agent.add_intersection_agent(agent_id)

        assert agent_id in agent.intersection_agents
        assert isinstance(agent.intersection_agents[agent_id], IntersectionAgent)
        assert agent_id in agent.intersection_hidden_states

    def test_add_district_agent(self, test_config):
        """Test adding district agent."""
        agent = HierarchicalTrafficAgent(test_config)

        agent_id = "district_test_1"
        agent.add_district_agent(agent_id)

        assert agent_id in agent.district_agents
        assert isinstance(agent.district_agents[agent_id], DistrictAgent)

    def test_forward_pass_intersection_only(self, test_config):
        """Test forward pass with only intersection agents."""
        agent = HierarchicalTrafficAgent(test_config)

        # Add intersection agents
        agent.add_intersection_agent("intersection_1")
        agent.add_intersection_agent("intersection_2")

        batch_size = 1
        obs_dim = test_config.agents.low_level.observation_space_dim

        observations = {
            "intersection_1": torch.randn(batch_size, obs_dim),
            "intersection_2": torch.randn(batch_size, obs_dim),
        }

        adjacency_matrices = {}
        agent_mappings = {}

        outputs = agent(observations, adjacency_matrices, agent_mappings)

        assert "intersection_1" in outputs
        assert "intersection_2" in outputs

        # Check output format for intersection agents
        for agent_id in ["intersection_1", "intersection_2"]:
            action_logits, values = outputs[agent_id]
            assert action_logits.shape == (batch_size, test_config.agents.low_level.action_space_size)
            assert values.shape == (batch_size, 1)

    def test_forward_pass_with_coordination(self, test_config):
        """Test forward pass with district coordination."""
        agent = HierarchicalTrafficAgent(test_config)

        # Add agents
        agent.add_intersection_agent("intersection_1")
        agent.add_intersection_agent("intersection_2")
        agent.add_district_agent("district_1")

        batch_size = 1
        obs_dim_low = test_config.agents.low_level.observation_space_dim
        obs_dim_high = test_config.agents.high_level.observation_space_dim

        observations = {
            "intersection_1": torch.randn(batch_size, obs_dim_low),
            "intersection_2": torch.randn(batch_size, obs_dim_low),
            "district_1": torch.randn(batch_size, obs_dim_high),
        }

        adjacency_matrices = {
            "district_1": torch.eye(2)  # 2 intersections
        }

        agent_mappings = {
            "district_1": ["intersection_1", "intersection_2"]
        }

        # Set coordination frequency to 1 for immediate coordination
        agent.coordination_frequency = 1

        outputs = agent(observations, adjacency_matrices, agent_mappings)

        assert "intersection_1" in outputs
        assert "intersection_2" in outputs
        assert "district_1" in outputs

    def test_reset_hidden_states(self, test_config):
        """Test hidden state reset."""
        agent = HierarchicalTrafficAgent(test_config)

        agent.add_intersection_agent("intersection_1")

        # Set some hidden state
        agent.intersection_hidden_states["intersection_1"] = (
            torch.randn(1, 1, 32),
            torch.randn(1, 1, 32)
        )

        assert agent.intersection_hidden_states["intersection_1"] is not None

        agent.reset_hidden_states()

        assert agent.intersection_hidden_states["intersection_1"] is None

    def test_save_and_load_checkpoint(self, test_config, temp_directory):
        """Test checkpoint saving and loading."""
        agent = HierarchicalTrafficAgent(test_config)

        agent.add_intersection_agent("intersection_1")
        agent.add_district_agent("district_1")

        checkpoint_path = temp_directory / "test_checkpoint.pth"

        # Save checkpoint
        agent.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()

        # Create new agent and load checkpoint
        new_agent = HierarchicalTrafficAgent(test_config)
        new_agent.add_intersection_agent("intersection_1")
        new_agent.add_district_agent("district_1")

        new_agent.load_checkpoint(str(checkpoint_path))

        # Verify loading worked (step count should match)
        assert new_agent.step_count == agent.step_count

    def test_get_agents_methods(self, test_config):
        """Test agent getter methods."""
        agent = HierarchicalTrafficAgent(test_config)

        agent.add_intersection_agent("intersection_1")
        agent.add_district_agent("district_1")

        intersection_agents = agent.get_intersection_agents()
        district_agents = agent.get_district_agents()

        assert isinstance(intersection_agents, dict)
        assert isinstance(district_agents, dict)
        assert "intersection_1" in intersection_agents
        assert "district_1" in district_agents


class TestTrafficEnvironmentWrapper:
    """Test cases for TrafficEnvironmentWrapper."""

    def test_initialization(self, test_config):
        """Test TrafficEnvironmentWrapper initialization."""
        hierarchical_agent = HierarchicalTrafficAgent(test_config)
        wrapper = TrafficEnvironmentWrapper(test_config, hierarchical_agent)

        assert wrapper.config == test_config
        assert wrapper.hierarchical_agent == hierarchical_agent
        assert isinstance(wrapper.device, torch.device)

    def test_predict_intersection_only(self, test_config, sample_observations):
        """Test prediction with intersection agents only."""
        hierarchical_agent = HierarchicalTrafficAgent(test_config)
        hierarchical_agent.add_intersection_agent("intersection_1")
        hierarchical_agent.add_intersection_agent("intersection_2")

        wrapper = TrafficEnvironmentWrapper(test_config, hierarchical_agent)

        observations = {
            "intersection_1": sample_observations["intersection_1"],
            "intersection_2": sample_observations["intersection_2"],
        }

        actions = wrapper.predict(observations, deterministic=True)

        assert "intersection_1" in actions
        assert "intersection_2" in actions

        # Check action format (discrete for intersection agents)
        for agent_id in ["intersection_1", "intersection_2"]:
            assert isinstance(actions[agent_id], np.ndarray)
            assert actions[agent_id].shape == ()  # Scalar action

    def test_predict_with_district_agents(self, test_config, sample_observations):
        """Test prediction with district agents."""
        hierarchical_agent = HierarchicalTrafficAgent(test_config)
        hierarchical_agent.add_intersection_agent("intersection_1")
        hierarchical_agent.add_district_agent("district_1")

        wrapper = TrafficEnvironmentWrapper(test_config, hierarchical_agent)

        observations = {
            "intersection_1": sample_observations["intersection_1"],
            "district_1": sample_observations["district_1"],
        }

        actions = wrapper.predict(observations, deterministic=False)

        assert "intersection_1" in actions
        assert "district_1" in actions

        # Check action formats
        assert isinstance(actions["intersection_1"], np.ndarray)
        assert isinstance(actions["district_1"], np.ndarray)

        # District actions should be continuous
        assert actions["district_1"].shape == (test_config.agents.high_level.action_space_size,)

    def test_create_agent_mappings(self, test_config):
        """Test agent mapping creation."""
        hierarchical_agent = HierarchicalTrafficAgent(test_config)
        hierarchical_agent.add_intersection_agent("intersection_1")
        hierarchical_agent.add_intersection_agent("intersection_2")
        hierarchical_agent.add_district_agent("district_1")

        wrapper = TrafficEnvironmentWrapper(test_config, hierarchical_agent)

        mappings = wrapper._create_agent_mappings()

        assert isinstance(mappings, dict)
        assert "district_1" in mappings
        assert isinstance(mappings["district_1"], list)

    @patch('torch.cuda.is_available')
    def test_device_selection_cpu(self, mock_cuda, test_config):
        """Test CPU device selection."""
        mock_cuda.return_value = False

        hierarchical_agent = HierarchicalTrafficAgent(test_config)
        wrapper = TrafficEnvironmentWrapper(test_config, hierarchical_agent)

        assert wrapper.device.type == "cpu"

    @patch('torch.cuda.is_available')
    def test_device_selection_gpu(self, mock_cuda, test_config):
        """Test GPU device selection."""
        mock_cuda.return_value = True

        hierarchical_agent = HierarchicalTrafficAgent(test_config)
        wrapper = TrafficEnvironmentWrapper(test_config, hierarchical_agent)

        assert wrapper.device.type in ["cuda", "cpu"]  # May fall back to CPU in test environment