"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import numpy as np

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.utils.config import Config


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Create test configuration.

    Returns:
        Test configuration object.
    """
    config_dict = {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "log_level": "WARNING",
            "output_dir": "test_outputs"
        },
        "environment": {
            "type": "sumo",
            "scenario": "manhattan_grid",
            "grid_size": [3, 3],
            "simulation_time": 300,
            "step_length": 1.0,
            "yellow_duration": 3,
            "min_green_duration": 5,
            "max_green_duration": 30,
            "observation": {
                "vehicle_positions": True,
                "queue_lengths": True,
                "waiting_times": True,
                "phase_durations": True,
                "neighboring_states": True,
                "history_length": 2
            },
            "reward": {
                "waiting_time_weight": -1.0,
                "throughput_weight": 0.5,
                "coordination_weight": 0.3,
                "fuel_consumption_weight": -0.2
            }
        },
        "agents": {
            "hierarchy_levels": 2,
            "low_level": {
                "type": "intersection_agent",
                "algorithm": "PPO",
                "observation_space_dim": 32,
                "action_space_type": "discrete",
                "action_space_size": 4
            },
            "high_level": {
                "type": "district_agent",
                "algorithm": "SAC",
                "observation_space_dim": 64,
                "action_space_type": "continuous",
                "action_space_size": 8,
                "district_size": [2, 2]
            }
        },
        "training": {
            "framework": "sb3",
            "total_timesteps": 1000,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coeff": 0.01,
            "value_function_coeff": 0.5,
            "max_grad_norm": 0.5,
            "hierarchical": {
                "pretrain_low_level": False,
                "pretrain_steps": 500,
                "coordination_frequency": 5,
                "communication_dim": 16
            },
            "multi_agent": {
                "shared_policy": False,
                "parameter_sharing": False,
                "centralized_critic": False
            },
            "curriculum": {
                "enabled": False,
                "initial_difficulty": 0.3,
                "max_difficulty": 1.0,
                "difficulty_increment": 0.1,
                "performance_threshold": 0.8
            }
        },
        "model": {
            "low_level_net": {
                "hidden_layers": [64, 64],
                "activation": "relu",
                "use_lstm": True,
                "lstm_hidden_size": 32
            },
            "high_level_net": {
                "hidden_layers": [128, 128],
                "activation": "relu",
                "attention_heads": 4,
                "use_transformer": True
            },
            "communication_net": {
                "hidden_layers": [32, 16],
                "activation": "tanh"
            }
        },
        "evaluation": {
            "interval": 500,
            "episodes": 3,
            "metrics": [
                "average_wait_time",
                "throughput",
                "coordination_efficiency"
            ],
            "transfer_scenarios": [
                "manhattan_to_cologne"
            ]
        },
        "logging": {
            "mlflow": {
                "enabled": False,
                "experiment_name": "test_experiments",
                "tracking_uri": "file:./test_mlruns"
            },
            "checkpoints": {
                "enabled": False,
                "frequency": 1000,
                "keep_last": 2
            },
            "tensorboard": {
                "enabled": False,
                "log_dir": "test_logs/tensorboard"
            }
        },
        "data": {
            "sumo": {
                "binary_path": "/usr/bin/sumo",
                "config_dir": "test_data/sumo_configs"
            },
            "datasets": {
                "cologne": {
                    "url": "test_url",
                    "local_path": "test_data/cologne"
                },
                "manhattan": {
                    "generate_synthetic": True,
                    "grid_sizes": [[3, 3]],
                    "traffic_densities": [0.5]
                }
            }
        },
        "target_metrics": {
            "average_wait_time_reduction": 0.20,
            "throughput_improvement": 0.15,
            "coordination_efficiency": 0.60,
            "transfer_learning_retention": 0.50
        }
    }

    return Config(config_dict)


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_intersection_data() -> Dict[str, Any]:
    """Create sample intersection data.

    Returns:
        Sample intersection configuration.
    """
    return {
        "id": "test_intersection_1",
        "coord": (100.0, 200.0),
        "incoming_edges": ["edge1", "edge2", "edge3", "edge4"],
        "outgoing_edges": ["edge5", "edge6", "edge7", "edge8"],
        "tls_id": "tls_test_1"
    }


@pytest.fixture
def sample_network_topology():
    """Create sample network topology.

    Returns:
        Sample network topology data.
    """
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes (intersections)
    intersections = [
        ("intersection_1", {"coord": (0, 0), "type": "intersection"}),
        ("intersection_2", {"coord": (100, 0), "type": "intersection"}),
        ("intersection_3", {"coord": (0, 100), "type": "intersection"}),
        ("intersection_4", {"coord": (100, 100), "type": "intersection"}),
    ]

    for node_id, attrs in intersections:
        G.add_node(node_id, **attrs)

    # Add edges (roads)
    roads = [
        ("intersection_1", "intersection_2", {"length": 100, "speed_limit": 50}),
        ("intersection_2", "intersection_1", {"length": 100, "speed_limit": 50}),
        ("intersection_1", "intersection_3", {"length": 100, "speed_limit": 50}),
        ("intersection_3", "intersection_1", {"length": 100, "speed_limit": 50}),
        ("intersection_2", "intersection_4", {"length": 100, "speed_limit": 50}),
        ("intersection_4", "intersection_2", {"length": 100, "speed_limit": 50}),
        ("intersection_3", "intersection_4", {"length": 100, "speed_limit": 50}),
        ("intersection_4", "intersection_3", {"length": 100, "speed_limit": 50}),
    ]

    for src, dst, attrs in roads:
        G.add_edge(src, dst, **attrs)

    return G


@pytest.fixture
def sample_observations() -> Dict[str, np.ndarray]:
    """Create sample agent observations.

    Returns:
        Dictionary of sample observations.
    """
    return {
        "intersection_1": np.random.randn(32).astype(np.float32),
        "intersection_2": np.random.randn(32).astype(np.float32),
        "district_1": np.random.randn(64).astype(np.float32),
    }


@pytest.fixture
def sample_vehicle_data() -> Dict[str, Any]:
    """Create sample vehicle data.

    Returns:
        Sample vehicle data dictionary.
    """
    return {
        "vehicles": {
            "lane_counts": {
                "lane_1": 5,
                "lane_2": 3,
                "lane_3": 7,
                "lane_4": 2
            },
            "lane_speeds": {
                "lane_1": [10.5, 12.3, 8.9, 11.2, 9.8],
                "lane_2": [15.2, 13.7, 14.1],
                "lane_3": [0.5, 0.2, 0.8, 1.2, 0.0, 0.3, 0.7],
                "lane_4": [20.1, 18.9]
            },
            "lane_densities": {
                "lane_1": 0.025,
                "lane_2": 0.015,
                "lane_3": 0.035,
                "lane_4": 0.010
            }
        },
        "queue_lengths": {
            "lane_queues": {
                "lane_1": 2,
                "lane_2": 1,
                "lane_3": 5,
                "lane_4": 0
            },
            "queue_rates": {
                "lane_1": 0.1,
                "lane_2": -0.2,
                "lane_3": 0.3,
                "lane_4": 0.0
            },
            "max_queue_normalized": {
                "lane_1": 0.4,
                "lane_2": 0.2,
                "lane_3": 0.8,
                "lane_4": 0.0
            }
        },
        "waiting_times": {
            "lane_waiting_times": {
                "lane_1": [5.2, 3.1],
                "lane_2": [12.5],
                "lane_3": [25.3, 18.7, 30.2, 22.1, 28.9],
                "lane_4": []
            },
            "max_waiting_times": {
                "lane_1": 5.2,
                "lane_2": 12.5,
                "lane_3": 30.2,
                "lane_4": 0.0
            },
            "cumulative_waiting": {
                "lane_1": 8.3,
                "lane_2": 12.5,
                "lane_3": 125.2,
                "lane_4": 0.0
            }
        },
        "tls_phase": {
            "current_phase": 1,
            "num_phases": 4,
            "urgency_scores": [0.3, 0.7, 0.2, 0.1],
            "phase_duration": 15.0
        },
        "neighbors": {
            "neighbors": [
                {
                    "total_queue_length": 8.5,
                    "throughput": 3.2,
                    "distance": 150.0,
                    "coordination_signal": 0.4
                },
                {
                    "total_queue_length": 12.1,
                    "throughput": 2.8,
                    "distance": 200.0,
                    "coordination_signal": -0.2
                }
            ]
        }
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_sumo_environment(monkeypatch):
    """Mock SUMO environment for testing without actual simulation.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    class MockTraci:
        """Mock SUMO TraCI interface."""

        @staticmethod
        def start(cmd, label=None):
            pass

        @staticmethod
        def switch(label):
            pass

        @staticmethod
        def close():
            pass

        @staticmethod
        def simulationStep():
            pass

        class vehicle:
            @staticmethod
            def getIDList():
                return ["veh_1", "veh_2", "veh_3"]

            @staticmethod
            def getSpeed(veh_id):
                return 10.0 if veh_id in ["veh_1", "veh_2"] else 0.5

            @staticmethod
            def getWaitingTime(veh_id):
                return 5.0 if veh_id == "veh_3" else 0.0

        class lane:
            @staticmethod
            def getLastStepVehicleNumber(lane_id):
                return 3 if "lane" in lane_id else 0

            @staticmethod
            def getLastStepVehicleIDs(lane_id):
                return ["veh_1", "veh_2"] if "lane" in lane_id else []

            @staticmethod
            def getLength(lane_id):
                return 200.0

        class edge:
            @staticmethod
            def getIDList():
                return ["edge1", "edge2", "edge3"]

            @staticmethod
            def getLaneNumber(edge_id):
                return 2 if edge_id in ["edge1", "edge2"] else 1

        class trafficlight:
            @staticmethod
            def getIDList():
                return ["tls_1", "tls_2"]

            @staticmethod
            def getPhase(tls_id):
                return 1

            @staticmethod
            def setPhase(tls_id, phase):
                pass

            @staticmethod
            def getProgram(tls_id):
                return "default"

            @staticmethod
            def getPhaseDuration(tls_id):
                return 30.0

            @staticmethod
            def setPhaseDuration(tls_id, duration):
                pass

        class simulation:
            @staticmethod
            def getDepartedNumber():
                return 2

            @staticmethod
            def getArrivedNumber():
                return 1

    # Patch traci module
    import adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.data.loader as loader_module
    import adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer as trainer_module

    monkeypatch.setattr(loader_module, "traci", MockTraci)
    monkeypatch.setattr(trainer_module, "traci", MockTraci)

    # Mock subprocess calls for SUMO tools
    def mock_subprocess_run(*args, **kwargs):
        class MockResult:
            returncode = 0
            stdout = "Mock SUMO output"
            stderr = ""

        return MockResult()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)


@pytest.fixture
def sample_training_metrics() -> List[Dict[str, float]]:
    """Create sample training metrics for testing.

    Returns:
        List of sample metrics dictionaries.
    """
    metrics = []
    for i in range(10):
        episode_metrics = {
            "average_wait_time": 30.0 + np.random.normal(0, 5),
            "throughput_rate": 2.0 + np.random.normal(0, 0.5),
            "coordination_efficiency": 0.6 + np.random.normal(0, 0.1),
            "total_fuel_consumption": 100.0 + np.random.normal(0, 10),
            "average_queue_length": 5.0 + np.random.normal(0, 1),
            "max_queue_length": 15.0 + np.random.normal(0, 2),
        }
        metrics.append(episode_metrics)

    return metrics


@pytest.fixture
def sample_baseline_metrics() -> Dict[str, float]:
    """Create sample baseline metrics.

    Returns:
        Sample baseline metrics dictionary.
    """
    return {
        "average_wait_time": 45.0,
        "average_wait_time_std": 8.0,
        "throughput_rate": 1.5,
        "throughput_rate_std": 0.3,
        "coordination_efficiency": 0.2,
        "total_fuel_consumption": 150.0,
        "average_queue_length": 8.0,
        "max_queue_length": 20.0,
    }


@pytest.fixture
def cleanup_test_files():
    """Clean up test files after tests.

    Yields:
        Cleanup function.
    """
    created_files = []
    created_dirs = []

    def track_file(filepath):
        created_files.append(filepath)

    def track_dir(dirpath):
        created_dirs.append(dirpath)

    yield track_file, track_dir

    # Cleanup
    for filepath in created_files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

    for dirpath in created_dirs:
        if os.path.exists(dirpath):
            try:
                os.rmdir(dirpath)
            except Exception:
                pass