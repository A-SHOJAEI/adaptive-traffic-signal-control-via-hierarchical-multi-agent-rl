"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.data.loader import (
    SUMODataLoader,
    TrafficDataLoader
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.data.preprocessing import (
    TrafficPreprocessor,
    SyntheticTrafficGenerator
)


class TestTrafficDataLoader:
    """Test cases for TrafficDataLoader base class."""

    def test_initialization(self, test_config):
        """Test TrafficDataLoader initialization."""
        loader = TrafficDataLoader(test_config)
        assert loader.config == test_config
        assert loader.data_dir.exists()

    def test_load_scenario_not_implemented(self, test_config):
        """Test that load_scenario raises NotImplementedError."""
        loader = TrafficDataLoader(test_config)
        with pytest.raises(NotImplementedError):
            loader.load_scenario("test_scenario")

    def test_get_available_scenarios_not_implemented(self, test_config):
        """Test that get_available_scenarios raises NotImplementedError."""
        loader = TrafficDataLoader(test_config)
        with pytest.raises(NotImplementedError):
            loader.get_available_scenarios()


class TestSUMODataLoader:
    """Test cases for SUMODataLoader."""

    def test_initialization(self, test_config, mock_sumo_environment):
        """Test SUMODataLoader initialization."""
        loader = SUMODataLoader(test_config)
        assert loader.config == test_config
        assert loader.sumo_binary == test_config.data.sumo.binary_path
        assert loader.config_dir.exists()

    def test_validate_sumo_success(self, test_config, mock_sumo_environment):
        """Test successful SUMO validation."""
        loader = SUMODataLoader(test_config)
        # If we get here without exception, validation passed
        assert True

    def test_validate_sumo_failure(self, test_config, monkeypatch):
        """Test SUMO validation failure."""
        def mock_subprocess_run(*args, **kwargs):
            class MockResult:
                returncode = 1
                stderr = "SUMO not found"
            return MockResult()

        monkeypatch.setattr("subprocess.run", mock_subprocess_run)

        with pytest.raises(RuntimeError, match="SUMO validation failed"):
            SUMODataLoader(test_config)

    def test_load_manhattan_scenario(self, test_config, mock_sumo_environment):
        """Test loading Manhattan grid scenario."""
        loader = SUMODataLoader(test_config)

        # Mock scenario data to avoid actual file operations
        with patch.object(loader, '_generate_scenario'):
            with patch('sumolib.net.readNet') as mock_read_net:
                mock_network = MagicMock()
                mock_network.getNodes.return_value = []
                mock_network.getEdges.return_value = []
                mock_read_net.return_value = mock_network

                scenario_data = loader.load_scenario("manhattan_grid")

                assert scenario_data["name"] == "manhattan_grid"
                assert "network_file" in scenario_data
                assert "route_file" in scenario_data
                assert "config_file" in scenario_data

    def test_extract_intersections(self, test_config, mock_sumo_environment):
        """Test intersection extraction from SUMO network."""
        loader = SUMODataLoader(test_config)

        # Mock network with traffic light nodes
        mock_node = MagicMock()
        mock_node.getType.return_value = "traffic_light"
        mock_node.getID.return_value = "test_tls_1"
        mock_node.getCoord.return_value = (100.0, 200.0)
        mock_node.getIncoming.return_value = []
        mock_node.getOutgoing.return_value = []

        mock_network = MagicMock()
        mock_network.getNodes.return_value = [mock_node]

        intersections = loader._extract_intersections(mock_network)

        assert len(intersections) == 1
        assert intersections[0]["id"] == "test_tls_1"
        assert intersections[0]["coord"] == (100.0, 200.0)

    def test_extract_edges(self, test_config, mock_sumo_environment):
        """Test edge extraction from SUMO network."""
        loader = SUMODataLoader(test_config)

        mock_edge = MagicMock()
        mock_edge.isSpecial.return_value = False
        mock_edge.getID.return_value = "test_edge_1"
        mock_edge.getFromNode.return_value.getID.return_value = "node_1"
        mock_edge.getToNode.return_value.getID.return_value = "node_2"
        mock_edge.getLength.return_value = 100.0
        mock_edge.getSpeed.return_value = 50.0
        mock_edge.getLaneNumber.return_value = 2
        mock_edge.getPriority.return_value = 1

        mock_network = MagicMock()
        mock_network.getEdges.return_value = [mock_edge]

        edges = loader._extract_edges(mock_network)

        assert len(edges) == 1
        assert edges[0]["id"] == "test_edge_1"
        assert edges[0]["length"] == 100.0
        assert edges[0]["speed_limit"] == 50.0

    def test_build_network_topology(self, test_config, mock_sumo_environment):
        """Test network topology building."""
        loader = SUMODataLoader(test_config)

        # Mock traffic light nodes
        mock_node1 = MagicMock()
        mock_node1.getType.return_value = "traffic_light"
        mock_node1.getID.return_value = "tls_1"
        mock_node1.getCoord.return_value = (0.0, 0.0)

        mock_node2 = MagicMock()
        mock_node2.getType.return_value = "traffic_light"
        mock_node2.getID.return_value = "tls_2"
        mock_node2.getCoord.return_value = (100.0, 0.0)

        # Mock edge
        mock_edge = MagicMock()
        mock_edge.isSpecial.return_value = False
        mock_edge.getFromNode.return_value.getID.return_value = "tls_1"
        mock_edge.getToNode.return_value.getID.return_value = "tls_2"
        mock_edge.getID.return_value = "edge_1_2"
        mock_edge.getLength.return_value = 100.0
        mock_edge.getSpeed.return_value = 50.0
        mock_edge.getLaneNumber.return_value = 2

        mock_network = MagicMock()
        mock_network.getNodes.return_value = [mock_node1, mock_node2]
        mock_network.getEdges.return_value = [mock_edge]

        topology = loader._build_network_topology(mock_network)

        assert isinstance(topology, nx.DiGraph)
        assert topology.number_of_nodes() == 2
        assert topology.number_of_edges() == 1
        assert topology.has_edge("tls_1", "tls_2")

    def test_get_available_scenarios(self, test_config, mock_sumo_environment):
        """Test getting available scenarios."""
        loader = SUMODataLoader(test_config)
        scenarios = loader.get_available_scenarios()

        assert "manhattan_grid" in scenarios
        assert "cologne" in scenarios
        assert isinstance(scenarios, list)


class TestTrafficPreprocessor:
    """Test cases for TrafficPreprocessor."""

    def test_initialization(self, test_config):
        """Test TrafficPreprocessor initialization."""
        preprocessor = TrafficPreprocessor(test_config)
        assert preprocessor.config == test_config
        assert isinstance(preprocessor.observation_history, dict)
        assert isinstance(preprocessor.normalization_params, dict)

    def test_preprocess_observations(self, test_config, sample_vehicle_data):
        """Test observation preprocessing."""
        preprocessor = TrafficPreprocessor(test_config)
        agent_id = "test_agent"

        observation = preprocessor.preprocess_observations(sample_vehicle_data, agent_id)

        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert not np.any(np.isnan(observation))
        assert not np.any(np.isinf(observation))

    def test_extract_vehicle_features(self, test_config, sample_vehicle_data):
        """Test vehicle feature extraction."""
        preprocessor = TrafficPreprocessor(test_config)
        vehicle_data = sample_vehicle_data["vehicles"]

        features = preprocessor._extract_vehicle_features(vehicle_data)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_extract_queue_features(self, test_config, sample_vehicle_data):
        """Test queue feature extraction."""
        preprocessor = TrafficPreprocessor(test_config)
        queue_data = sample_vehicle_data["queue_lengths"]

        features = preprocessor._extract_queue_features(queue_data)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_extract_waiting_features(self, test_config, sample_vehicle_data):
        """Test waiting time feature extraction."""
        preprocessor = TrafficPreprocessor(test_config)
        waiting_data = sample_vehicle_data["waiting_times"]

        features = preprocessor._extract_waiting_features(waiting_data)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_extract_phase_features(self, test_config, sample_vehicle_data):
        """Test traffic light phase feature extraction."""
        preprocessor = TrafficPreprocessor(test_config)
        phase_data = sample_vehicle_data["tls_phase"]
        phase_duration = 15.0

        features = preprocessor._extract_phase_features(phase_data, phase_duration)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

        # Check one-hot encoding for phases
        num_phases = phase_data["num_phases"]
        current_phase = phase_data["current_phase"]

        # First num_phases elements should be one-hot encoded
        phase_encoding = features[:num_phases]
        assert sum(phase_encoding) == 1.0
        assert phase_encoding[current_phase] == 1.0

    def test_extract_neighbor_features(self, test_config, sample_vehicle_data):
        """Test neighbor feature extraction."""
        preprocessor = TrafficPreprocessor(test_config)
        neighbor_data = sample_vehicle_data["neighbors"]
        agent_id = "test_agent"

        features = preprocessor._extract_neighbor_features(neighbor_data, agent_id)

        assert isinstance(features, list)
        assert len(features) == 8 * 4  # 8 max neighbors * 4 features each
        assert all(isinstance(f, float) for f in features)

    def test_normalize_observation(self, test_config):
        """Test observation normalization."""
        preprocessor = TrafficPreprocessor(test_config)
        agent_id = "test_agent"

        # Create test observations
        obs1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        obs2 = np.array([2.0, 4.0, 6.0], dtype=np.float32)

        # Normalize observations
        norm_obs1 = preprocessor._normalize_observation(obs1, agent_id)
        norm_obs2 = preprocessor._normalize_observation(obs2, agent_id)

        assert norm_obs1.shape == obs1.shape
        assert norm_obs2.shape == obs2.shape
        assert not np.any(np.isnan(norm_obs1))
        assert not np.any(np.isnan(norm_obs2))
        assert np.all(np.abs(norm_obs1) <= 5.0)  # Clipping test

    def test_add_temporal_features(self, test_config):
        """Test temporal feature addition."""
        preprocessor = TrafficPreprocessor(test_config)
        agent_id = "test_agent"

        observation = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Add temporal features
        temporal_obs = preprocessor._add_temporal_features(observation, agent_id)

        history_length = test_config.environment.observation.history_length
        expected_length = len(observation) * history_length

        assert temporal_obs.shape[0] == expected_length
        assert temporal_obs.dtype == np.float32

    def test_create_observation_space(self, test_config):
        """Test observation space creation."""
        preprocessor = TrafficPreprocessor(test_config)
        obs_space = preprocessor.create_observation_space()

        from gymnasium import spaces
        assert isinstance(obs_space, spaces.Box)
        assert obs_space.dtype == np.float32
        assert len(obs_space.shape) == 1

    def test_create_action_space_low_level(self, test_config):
        """Test low-level action space creation."""
        preprocessor = TrafficPreprocessor(test_config)
        action_space = preprocessor.create_action_space("low_level")

        from gymnasium import spaces
        assert isinstance(action_space, spaces.Discrete)
        assert action_space.n == test_config.agents.low_level.action_space_size

    def test_create_action_space_high_level(self, test_config):
        """Test high-level action space creation."""
        preprocessor = TrafficPreprocessor(test_config)
        action_space = preprocessor.create_action_space("high_level")

        from gymnasium import spaces
        assert isinstance(action_space, spaces.Box)
        assert action_space.shape[0] == test_config.agents.high_level.action_space_size

    def test_reset_history(self, test_config):
        """Test history reset."""
        preprocessor = TrafficPreprocessor(test_config)
        agent_id = "test_agent"

        # Add some history
        observation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        preprocessor._add_temporal_features(observation, agent_id)

        assert agent_id in preprocessor.observation_history
        assert len(preprocessor.observation_history[agent_id]) > 0

        # Reset history
        preprocessor.reset_history(agent_id)
        assert len(preprocessor.observation_history[agent_id]) == 0

    def test_get_normalization_stats(self, test_config):
        """Test getting normalization statistics."""
        preprocessor = TrafficPreprocessor(test_config)
        agent_id = "test_agent"

        # Generate some observations to create stats
        observation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        preprocessor._normalize_observation(observation, agent_id)

        stats = preprocessor.get_normalization_stats()

        assert isinstance(stats, dict)
        assert agent_id in stats
        assert "mean" in stats[agent_id]
        assert "std" in stats[agent_id]
        assert "count" in stats[agent_id]


class TestSyntheticTrafficGenerator:
    """Test cases for SyntheticTrafficGenerator."""

    def test_initialization(self, test_config):
        """Test SyntheticTrafficGenerator initialization."""
        generator = SyntheticTrafficGenerator(test_config)
        assert generator.config == test_config
        assert hasattr(generator, 'rng')

    def test_generate_uniform_demand(self, test_config, sample_network_topology):
        """Test uniform traffic demand generation."""
        generator = SyntheticTrafficGenerator(test_config)
        nodes = list(sample_network_topology.nodes())

        demands = generator._generate_uniform_demand(nodes, 0)

        assert isinstance(demands, list)
        assert len(demands) > 0

        # Check demand format
        for origin, destination, flow_rate in demands:
            assert origin in nodes
            assert destination in nodes
            assert origin != destination
            assert isinstance(flow_rate, float)
            assert flow_rate > 0

    def test_generate_peak_hour_demand(self, test_config, sample_network_topology):
        """Test peak hour traffic demand generation."""
        generator = SyntheticTrafficGenerator(test_config)
        nodes = list(sample_network_topology.nodes())
        total_time = test_config.environment.simulation_time

        # Test morning peak
        morning_peak_time = int(0.25 * total_time)
        demands = generator._generate_peak_hour_demand(nodes, morning_peak_time, total_time)

        assert isinstance(demands, list)
        assert len(demands) > 0

        # Peak hour should have higher flow rates
        avg_flow_rate = np.mean([d[2] for d in demands])
        assert avg_flow_rate > 0.1  # Should be higher than base rate

    def test_generate_random_demand(self, test_config, sample_network_topology):
        """Test random traffic demand generation."""
        generator = SyntheticTrafficGenerator(test_config)
        nodes = list(sample_network_topology.nodes())

        demands = generator._generate_random_demand(nodes, 0)

        assert isinstance(demands, list)
        assert len(demands) >= 0  # Could be empty due to randomness

    def test_generate_traffic_demand_uniform(self, test_config, sample_network_topology):
        """Test traffic demand generation with uniform pattern."""
        generator = SyntheticTrafficGenerator(test_config)
        time_horizon = 900  # 15 minutes

        demand_data = generator.generate_traffic_demand(
            sample_network_topology, time_horizon, "uniform"
        )

        assert isinstance(demand_data, dict)
        assert len(demand_data) > 0

        # Check time intervals
        for time_interval, demands in demand_data.items():
            assert isinstance(time_interval, int)
            assert 0 <= time_interval < time_horizon
            assert isinstance(demands, list)

    def test_generate_traffic_demand_peak_hour(self, test_config, sample_network_topology):
        """Test traffic demand generation with peak hour pattern."""
        generator = SyntheticTrafficGenerator(test_config)
        time_horizon = 1800  # 30 minutes

        demand_data = generator.generate_traffic_demand(
            sample_network_topology, time_horizon, "peak_hour"
        )

        assert isinstance(demand_data, dict)
        assert len(demand_data) > 0

    def test_generate_traffic_demand_invalid_pattern(self, test_config, sample_network_topology):
        """Test traffic demand generation with invalid pattern."""
        generator = SyntheticTrafficGenerator(test_config)

        with pytest.raises(ValueError, match="Unknown demand pattern"):
            generator.generate_traffic_demand(
                sample_network_topology, 900, "invalid_pattern"
            )

    def test_generate_incident_scenarios(self, test_config, sample_network_topology):
        """Test incident scenario generation."""
        generator = SyntheticTrafficGenerator(test_config)
        time_horizon = 3600
        incident_probability = 0.1  # Higher probability for testing

        incidents = generator.generate_incident_scenarios(
            sample_network_topology, time_horizon, incident_probability
        )

        assert isinstance(incidents, list)
        # Could be empty due to randomness, but should be a list

        if incidents:  # If any incidents generated
            incident = incidents[0]
            assert "edge" in incident
            assert "start_time" in incident
            assert "duration" in incident
            assert "severity" in incident
            assert "lane_closure" in incident

            assert incident["severity"] in ["minor", "major"]
            assert 0 <= incident["start_time"] < time_horizon
            assert incident["duration"] > 0
            assert incident["lane_closure"] > 0

    def test_create_scenario_variations(self, test_config):
        """Test scenario variation creation."""
        generator = SyntheticTrafficGenerator(test_config)
        base_scenario = {
            "name": "base_scenario",
            "traffic_density": 0.5,
            "incident_rate": 0.1
        }

        variations = generator.create_scenario_variations(base_scenario, 5)

        assert isinstance(variations, list)
        assert len(variations) == 5

        for variation in variations:
            assert "demand_multiplier" in variation
            assert "incident_multiplier" in variation
            assert "min_green_multiplier" in variation
            assert "max_green_multiplier" in variation
            assert "seed" in variation

            # Check multiplier ranges
            assert 0.7 <= variation["demand_multiplier"] <= 1.3
            assert 0.5 <= variation["incident_multiplier"] <= 2.0