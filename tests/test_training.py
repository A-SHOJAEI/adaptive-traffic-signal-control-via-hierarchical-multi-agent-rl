"""Tests for training modules."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
from gymnasium import spaces

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer import (
    TrafficEnvironment,
    HierarchicalTrainer,
    MLflowCallback
)


class TestTrafficEnvironment:
    """Test cases for TrafficEnvironment."""

    @pytest.fixture
    def env_config(self, test_config):
        """Create environment configuration."""
        return {
            "rl_config": test_config.to_dict(),
            "scenario": "manhattan_grid"
        }

    def test_initialization(self, env_config, mock_sumo_environment):
        """Test TrafficEnvironment initialization."""
        env = TrafficEnvironment(env_config)

        assert env.env_config == env_config
        assert isinstance(env.rl_config.experiment.name, str)
        assert env.scenario_name == "manhattan_grid"
        assert env.current_step == 0
        assert env.max_steps > 0
        assert isinstance(env.intersection_agents, dict)
        assert isinstance(env.district_agents, dict)

    def test_setup_agents(self, env_config, mock_sumo_environment):
        """Test agent setup."""
        env = TrafficEnvironment(env_config)

        # Should have created intersection and district agents
        assert len(env.intersection_agents) > 0 or len(env.district_agents) > 0

        # Check agent structure
        for agent_id, agent_data in env.intersection_agents.items():
            assert "intersection_data" in agent_data
            assert agent_data["type"] == "low_level"

        for agent_id, agent_data in env.district_agents.items():
            assert "district_coords" in agent_data
            assert agent_data["type"] == "high_level"
            assert "managed_intersections" in agent_data

    def test_setup_spaces(self, env_config, mock_sumo_environment):
        """Test observation and action space setup."""
        env = TrafficEnvironment(env_config)

        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.action_space, spaces.Dict)

        # Check that spaces match agent IDs
        assert set(env.observation_space.spaces.keys()) == env._agent_ids
        assert set(env.action_space.spaces.keys()) == env._agent_ids

    def test_reset(self, env_config, mock_sumo_environment):
        """Test environment reset."""
        env = TrafficEnvironment(env_config)

        observations, info = env.reset()

        assert isinstance(observations, dict)
        assert isinstance(info, dict)
        assert env.current_step == 0
        assert env.sumo_label is not None

        # Check observation format
        for agent_id in env._agent_ids:
            assert agent_id in observations
            assert isinstance(observations[agent_id], np.ndarray)

    def test_step(self, env_config, mock_sumo_environment):
        """Test environment step."""
        env = TrafficEnvironment(env_config)
        observations, _ = env.reset()

        # Create valid actions
        actions = {}
        for agent_id in env.intersection_agents:
            actions[agent_id] = 0  # Valid discrete action
        for agent_id in env.district_agents:
            action_size = env.action_space[agent_id].shape[0]
            actions[agent_id] = np.zeros(action_size)  # Valid continuous action

        # Step environment
        new_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        assert isinstance(new_obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminateds, dict)
        assert isinstance(truncateds, dict)
        assert isinstance(infos, dict)

        # Check that all agent IDs are present
        for agent_id in env._agent_ids:
            assert agent_id in new_obs
            assert agent_id in rewards
            assert agent_id in terminateds
            assert agent_id in truncateds
            assert agent_id in infos

        assert env.current_step == 1

    def test_apply_intersection_actions(self, env_config, mock_sumo_environment):
        """Test intersection action application."""
        env = TrafficEnvironment(env_config)
        env.reset()

        if env.intersection_agents:
            agent_id = list(env.intersection_agents.keys())[0]
            action = 2  # Valid phase action

            # This should not raise an exception
            env._apply_intersection_action(agent_id, action)

    def test_apply_district_actions(self, env_config, mock_sumo_environment):
        """Test district action application."""
        env = TrafficEnvironment(env_config)
        env.reset()

        if env.district_agents:
            agent_id = list(env.district_agents.keys())[0]
            action = np.array([0.5, -0.3, 0.1, 0.0])  # Valid continuous action

            # This should not raise an exception
            env._apply_district_action(agent_id, action)

    def test_get_observations(self, env_config, mock_sumo_environment):
        """Test observation collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        observations = env._get_observations()

        assert isinstance(observations, dict)
        for agent_id in env._agent_ids:
            assert agent_id in observations
            assert isinstance(observations[agent_id], np.ndarray)
            assert not np.any(np.isnan(observations[agent_id]))

    def test_get_intersection_observation(self, env_config, mock_sumo_environment):
        """Test intersection observation collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        if env.intersection_agents:
            agent_id = list(env.intersection_agents.keys())[0]
            observation = env._get_intersection_observation(agent_id)

            assert isinstance(observation, np.ndarray)
            assert not np.any(np.isnan(observation))
            assert not np.any(np.isinf(observation))

    def test_get_district_observation(self, env_config, mock_sumo_environment):
        """Test district observation collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        if env.district_agents:
            agent_id = list(env.district_agents.keys())[0]
            observation = env._get_district_observation(agent_id)

            assert isinstance(observation, np.ndarray)
            assert not np.any(np.isnan(observation))
            assert not np.any(np.isinf(observation))

    def test_calculate_rewards(self, env_config, mock_sumo_environment):
        """Test reward calculation."""
        env = TrafficEnvironment(env_config)
        env.reset()

        rewards = env._calculate_rewards()

        assert isinstance(rewards, dict)
        for agent_id in env._agent_ids:
            assert agent_id in rewards
            assert isinstance(rewards[agent_id], float)
            assert not np.isnan(rewards[agent_id])
            assert not np.isinf(rewards[agent_id])

    def test_calculate_intersection_reward(self, env_config, mock_sumo_environment):
        """Test intersection reward calculation."""
        env = TrafficEnvironment(env_config)
        env.reset()

        if env.intersection_agents:
            agent_id = list(env.intersection_agents.keys())[0]
            reward = env._calculate_intersection_reward(agent_id)

            assert isinstance(reward, float)
            assert not np.isnan(reward)
            assert not np.isinf(reward)

    def test_calculate_district_reward(self, env_config, mock_sumo_environment):
        """Test district reward calculation."""
        env = TrafficEnvironment(env_config)
        env.reset()

        if env.district_agents:
            agent_id = list(env.district_agents.keys())[0]
            reward = env._calculate_district_reward(agent_id)

            assert isinstance(reward, float)
            assert not np.isnan(reward)
            assert not np.isinf(reward)

    def test_update_metrics(self, env_config, mock_sumo_environment):
        """Test metrics update."""
        env = TrafficEnvironment(env_config)
        env.reset()

        initial_metrics = env.episode_metrics.copy()
        env._update_metrics()

        # Metrics should be updated (may be the same values due to mocking)
        assert isinstance(env.episode_metrics, dict)
        assert "total_waiting_time" in env.episode_metrics
        assert "total_throughput" in env.episode_metrics

    def test_get_vehicle_data(self, env_config, mock_sumo_environment, sample_intersection_data):
        """Test vehicle data collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        vehicle_data = env._get_vehicle_data(sample_intersection_data)

        assert isinstance(vehicle_data, dict)
        assert "lane_counts" in vehicle_data
        assert "lane_speeds" in vehicle_data
        assert "lane_densities" in vehicle_data

    def test_get_queue_data(self, env_config, mock_sumo_environment, sample_intersection_data):
        """Test queue data collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        queue_data = env._get_queue_data(sample_intersection_data)

        assert isinstance(queue_data, dict)
        assert "lane_queues" in queue_data
        assert "queue_rates" in queue_data
        assert "max_queue_normalized" in queue_data

    def test_get_waiting_data(self, env_config, mock_sumo_environment, sample_intersection_data):
        """Test waiting time data collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        waiting_data = env._get_waiting_data(sample_intersection_data)

        assert isinstance(waiting_data, dict)
        assert "lane_waiting_times" in waiting_data
        assert "max_waiting_times" in waiting_data
        assert "cumulative_waiting" in waiting_data

    def test_get_tls_data(self, env_config, mock_sumo_environment):
        """Test traffic light data collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        # Test with valid TLS ID
        tls_data = env._get_tls_data("tls_1")

        assert isinstance(tls_data, dict)
        assert "current_phase" in tls_data
        assert "num_phases" in tls_data
        assert "urgency_scores" in tls_data

        # Test with invalid TLS ID
        tls_data_invalid = env._get_tls_data("invalid_tls")
        assert isinstance(tls_data_invalid, dict)
        assert tls_data_invalid["current_phase"] == 0

    def test_get_neighbor_data(self, env_config, mock_sumo_environment):
        """Test neighbor data collection."""
        env = TrafficEnvironment(env_config)
        env.reset()

        neighbor_data = env._get_neighbor_data("test_agent")

        assert isinstance(neighbor_data, dict)
        assert "neighbors" in neighbor_data
        assert isinstance(neighbor_data["neighbors"], list)

    def test_get_terminateds(self, env_config, mock_sumo_environment):
        """Test termination conditions."""
        env = TrafficEnvironment(env_config)
        env.reset()

        terminateds = env._get_terminateds()

        assert isinstance(terminateds, dict)
        for agent_id in env._agent_ids:
            assert agent_id in terminateds
            assert isinstance(terminateds[agent_id], bool)

    def test_get_truncateds(self, env_config, mock_sumo_environment):
        """Test truncation conditions."""
        env = TrafficEnvironment(env_config)
        env.reset()

        truncateds = env._get_truncateds()

        assert isinstance(truncateds, dict)
        for agent_id in env._agent_ids:
            assert agent_id in truncateds
            assert isinstance(truncateds[agent_id], bool)
            assert not truncateds[agent_id]  # Should be False in this implementation

    def test_get_info(self, env_config, mock_sumo_environment):
        """Test info dictionary creation."""
        env = TrafficEnvironment(env_config)
        env.reset()

        info = env._get_info()

        assert isinstance(info, dict)
        for agent_id in env._agent_ids:
            assert agent_id in info
            assert isinstance(info[agent_id], dict)
            assert "step" in info[agent_id]
            assert "metrics" in info[agent_id]

    def test_close(self, env_config, mock_sumo_environment):
        """Test environment closure."""
        env = TrafficEnvironment(env_config)
        env.reset()

        # This should not raise an exception
        env.close()

        assert env.sumo_label is None

    def test_episode_completion(self, env_config, mock_sumo_environment):
        """Test complete episode run."""
        env = TrafficEnvironment(env_config)
        observations, _ = env.reset()

        # Run a few steps
        for _ in range(5):
            actions = {}
            for agent_id in env.intersection_agents:
                actions[agent_id] = 0
            for agent_id in env.district_agents:
                action_size = env.action_space[agent_id].shape[0]
                actions[agent_id] = np.zeros(action_size)

            observations, rewards, terminateds, truncateds, infos = env.step(actions)

            if all(terminateds.values()):
                break

        env.close()


class TestMLflowCallback:
    """Test cases for MLflowCallback."""

    def test_initialization(self):
        """Test MLflowCallback initialization."""
        callback = MLflowCallback(verbose=1)

        assert callback.verbose == 1
        assert isinstance(callback.episode_rewards, list)
        assert isinstance(callback.episode_lengths, list)

    def test_on_step_no_episode_info(self):
        """Test _on_step without episode information."""
        callback = MLflowCallback()
        callback.locals = {"infos": [{}]}
        callback.num_timesteps = 100

        # Should not raise an exception
        result = callback._on_step()
        assert result is True

    @patch('mlflow.log_metrics')
    def test_on_step_with_episode_info(self, mock_log_metrics):
        """Test _on_step with episode information."""
        callback = MLflowCallback()
        callback.locals = {
            "infos": [{
                "episode": {
                    "r": 50.0,
                    "l": 100
                }
            }]
        }
        callback.num_timesteps = 100

        result = callback._on_step()

        assert result is True
        assert len(callback.episode_rewards) == 1
        assert len(callback.episode_lengths) == 1
        assert callback.episode_rewards[0] == 50.0
        assert callback.episode_lengths[0] == 100

    @patch('mlflow.log_metrics')
    def test_on_step_logging_interval(self, mock_log_metrics):
        """Test logging at specific intervals."""
        callback = MLflowCallback()
        callback.locals = {"infos": [{}]}
        callback.num_timesteps = 1000
        callback.model = Mock()
        callback.model.learning_rate = 3e-4

        result = callback._on_step()

        assert result is True
        mock_log_metrics.assert_called()


class TestHierarchicalTrainer:
    """Test cases for HierarchicalTrainer."""

    def test_initialization_sb3(self, test_config):
        """Test HierarchicalTrainer initialization with SB3."""
        test_config.training.framework = "sb3"
        test_config.logging.mlflow.enabled = False

        trainer = HierarchicalTrainer(test_config)

        assert trainer.config == test_config
        assert trainer.framework == "sb3"

    @patch('ray.init')
    def test_initialization_ray(self, mock_ray_init, test_config):
        """Test HierarchicalTrainer initialization with Ray."""
        test_config.training.framework = "ray"
        test_config.logging.mlflow.enabled = False

        with patch('ray.is_initialized', return_value=False):
            trainer = HierarchicalTrainer(test_config)

            assert trainer.framework == "ray"
            mock_ray_init.assert_called_once()

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_train_sb3_framework(self, mock_log_params, mock_start_run, test_config, mock_sumo_environment):
        """Test training with SB3 framework."""
        test_config.training.framework = "sb3"
        test_config.logging.mlflow.enabled = True

        trainer = HierarchicalTrainer(test_config)

        # Mock the training methods
        with patch.object(trainer, '_train_with_sb3', return_value={"status": "completed"}):
            results = trainer.train()

            assert isinstance(results, dict)
            assert results["status"] == "completed"

    @patch('ray.tune.registry.register_env')
    @patch('mlflow.start_run')
    def test_train_ray_framework(self, mock_start_run, mock_register_env, test_config, mock_sumo_environment):
        """Test training with Ray framework."""
        test_config.training.framework = "ray"
        test_config.logging.mlflow.enabled = True

        trainer = HierarchicalTrainer(test_config)

        # Mock the training methods
        with patch.object(trainer, '_train_with_ray', return_value={"status": "completed"}):
            results = trainer.train()

            assert isinstance(results, dict)

    def test_pretrain_intersection_agents(self, test_config, mock_sumo_environment):
        """Test intersection agent pre-training."""
        test_config.training.framework = "sb3"
        trainer = HierarchicalTrainer(test_config)

        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import HierarchicalTrafficAgent
        hierarchical_agent = HierarchicalTrafficAgent(test_config)

        # Mock environment and model
        with patch('adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer.TrafficEnvironment'):
            with patch('stable_baselines3.PPO') as mock_ppo:
                with patch('stable_baselines3.common.evaluation.evaluate_policy', return_value=([10.0], [100])):
                    mock_model = Mock()
                    mock_model.learn = Mock()
                    mock_model.save = Mock()
                    mock_ppo.return_value = mock_model

                    results = trainer._pretrain_intersection_agents(hierarchical_agent)

                    assert isinstance(results, dict)
                    assert "mean_reward" in results
                    assert "mean_length" in results
                    assert "model_path" in results

    def test_train_district_agents(self, test_config, mock_sumo_environment):
        """Test district agent training."""
        test_config.training.framework = "sb3"
        trainer = HierarchicalTrainer(test_config)

        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import HierarchicalTrafficAgent
        hierarchical_agent = HierarchicalTrafficAgent(test_config)

        # Mock environment and model
        with patch('adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer.TrafficEnvironment'):
            with patch('stable_baselines3.SAC') as mock_sac:
                with patch('stable_baselines3.common.evaluation.evaluate_policy', return_value=([15.0], [120])):
                    mock_model = Mock()
                    mock_model.learn = Mock()
                    mock_model.save = Mock()
                    mock_sac.return_value = mock_model

                    results = trainer._train_district_agents(hierarchical_agent)

                    assert isinstance(results, dict)
                    assert "mean_reward" in results
                    assert "mean_length" in results
                    assert "model_path" in results

    def test_joint_fine_tuning(self, test_config, mock_sumo_environment):
        """Test joint fine-tuning."""
        trainer = HierarchicalTrainer(test_config)

        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import HierarchicalTrafficAgent
        hierarchical_agent = HierarchicalTrafficAgent(test_config)

        # Mock environment
        with patch('adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer.TrafficEnvironment'):
            results = trainer._joint_fine_tuning(hierarchical_agent)

            assert isinstance(results, dict)
            assert "message" in results

    def test_create_ray_policies(self, test_config):
        """Test Ray policy creation."""
        trainer = HierarchicalTrainer(test_config)

        policies = trainer._create_ray_policies()

        assert isinstance(policies, dict)
        assert "intersection_policy" in policies
        assert "district_policy" in policies

    def test_policy_mapping_fn(self, test_config):
        """Test policy mapping function."""
        trainer = HierarchicalTrainer(test_config)

        # Test intersection agent mapping
        intersection_policy = trainer._policy_mapping_fn("intersection_1")
        assert intersection_policy == "intersection_policy"

        # Test district agent mapping
        district_policy = trainer._policy_mapping_fn("district_1")
        assert district_policy == "district_policy"

        # Test default mapping
        default_policy = trainer._policy_mapping_fn("unknown_agent")
        assert default_policy == "intersection_policy"

    @patch('adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer.TrafficEnvironment')
    def test_evaluate_ray_model(self, mock_env_class, test_config, mock_sumo_environment):
        """Test Ray model evaluation."""
        trainer = HierarchicalTrainer(test_config)

        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = ({"agent_1": np.array([1, 2, 3])}, {})
        mock_env.step.return_value = (
            {"agent_1": np.array([1, 2, 3])},
            {"agent_1": 5.0},
            {"agent_1": True},
            {"agent_1": False},
            {"agent_1": {}}
        )
        mock_env.close = Mock()
        mock_env_class.return_value = mock_env

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.compute_single_action.return_value = 0

        results = trainer._evaluate_ray_model(mock_trainer)

        assert isinstance(results, dict)
        assert "mean_reward" in results
        assert "mean_length" in results

    def test_evaluate_transfer_learning(self, test_config):
        """Test transfer learning evaluation."""
        trainer = HierarchicalTrainer(test_config)

        results = trainer.evaluate_transfer_learning()

        assert isinstance(results, dict)
        for scenario_pair in test_config.evaluation.transfer_scenarios:
            assert scenario_pair in results
            assert "source_performance" in results[scenario_pair]
            assert "target_performance" in results[scenario_pair]
            assert "transfer_retention" in results[scenario_pair]

    @patch('mlflow.log_artifact')
    def test_save_final_model(self, mock_log_artifact, test_config):
        """Test final model saving."""
        trainer = HierarchicalTrainer(test_config)

        model_path = "test_model.pth"
        trainer.save_final_model(model_path)

        mock_log_artifact.assert_called_once_with(model_path)

    @patch('ray.shutdown')
    def test_cleanup_ray(self, mock_shutdown, test_config):
        """Test cleanup with Ray."""
        test_config.training.framework = "ray"

        with patch('ray.is_initialized', return_value=True):
            trainer = HierarchicalTrainer(test_config)
            trainer.cleanup()

            mock_shutdown.assert_called_once()

    def test_cleanup_sb3(self, test_config):
        """Test cleanup with SB3."""
        test_config.training.framework = "sb3"
        trainer = HierarchicalTrainer(test_config)

        # Should not raise an exception
        trainer.cleanup()

    def test_error_handling_invalid_framework(self, test_config):
        """Test error handling for invalid framework."""
        test_config.training.framework = "invalid_framework"
        trainer = HierarchicalTrainer(test_config)

        # The trainer should handle unknown frameworks gracefully
        # by falling back to SB3 implementation
        with patch.object(trainer, '_train_with_sb3', return_value={}):
            results = trainer.train()
            assert isinstance(results, dict)