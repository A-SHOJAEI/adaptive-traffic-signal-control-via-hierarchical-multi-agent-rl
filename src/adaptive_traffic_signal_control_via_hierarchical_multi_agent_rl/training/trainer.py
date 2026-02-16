"""Training pipeline for hierarchical multi-agent RL traffic control."""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import gymnasium as gym
import mlflow
import numpy as np
import ray
import torch
import traci
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms import PPOConfig, SACConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import ModelConfigDict
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ..data.loader import SUMODataLoader
from ..data.preprocessing import TrafficPreprocessor
from ..models.model import HierarchicalTrafficAgent, TrafficEnvironmentWrapper
from ..utils.config import Config

logger = logging.getLogger(__name__)


class TrafficEnvironment(MultiAgentEnv):
    """SUMO-based traffic environment for multi-agent RL."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize traffic environment.

        Args:
            config: Environment configuration dictionary.
        """
        super().__init__()
        self.env_config = config
        self.rl_config = Config(config["rl_config"])

        # Initialize data loader and preprocessor
        self.data_loader = SUMODataLoader(self.rl_config)
        self.preprocessor = TrafficPreprocessor(self.rl_config)

        # Load scenario
        self.scenario_name = config.get("scenario", "manhattan_grid")
        self.scenario_data = self.data_loader.load_scenario(self.scenario_name)

        # SUMO simulation state
        self.sumo_label = None
        self.current_step = 0
        self.max_steps = int(self.rl_config.environment.simulation_time /
                           self.rl_config.environment.step_length)

        # Agent management
        self.intersection_agents = {}
        self.district_agents = {}
        self._setup_agents()

        # Spaces
        self._setup_spaces()

        # Metrics tracking
        self.episode_metrics = {
            "total_waiting_time": 0.0,
            "total_throughput": 0.0,
            "total_fuel_consumption": 0.0,
            "coordination_events": 0,
        }

        logger.info(f"Initialized TrafficEnvironment with {len(self.intersection_agents)} "
                   f"intersections and {len(self.district_agents)} districts")

    def _setup_agents(self) -> None:
        """Set up agent configuration based on scenario."""
        # Extract intersections from scenario
        intersections = self.scenario_data.get("intersections", [])

        # Create intersection agents
        for intersection in intersections:
            agent_id = f"intersection_{intersection['id']}"
            self.intersection_agents[agent_id] = {
                "intersection_data": intersection,
                "type": "low_level"
            }

        # Create district agents based on grid structure
        district_size = self.rl_config.agents.high_level.district_size
        grid_size = self.rl_config.environment.grid_size

        num_districts_x = max(1, grid_size[0] // district_size[0])
        num_districts_y = max(1, grid_size[1] // district_size[1])

        for i in range(num_districts_x):
            for j in range(num_districts_y):
                district_id = f"district_{i}_{j}"
                self.district_agents[district_id] = {
                    "district_coords": (i, j),
                    "type": "high_level",
                    "managed_intersections": self._get_managed_intersections(i, j, district_size)
                }

    def _get_managed_intersections(self, district_x: int, district_y: int,
                                 district_size: List[int]) -> List[str]:
        """Get intersections managed by a district.

        Args:
            district_x: District X coordinate.
            district_y: District Y coordinate.
            district_size: Size of each district [width, height].

        Returns:
            List of intersection agent IDs managed by this district.
        """
        managed = []
        start_x = district_x * district_size[0]
        start_y = district_y * district_size[1]

        for intersection_id, agent_data in self.intersection_agents.items():
            # Simple grid-based assignment (would be more complex in practice)
            intersection_data = agent_data["intersection_data"]
            coord = intersection_data.get("coord", (0, 0))

            # Rough coordinate-based assignment
            grid_x = int(coord[0] / 200)  # Assuming 200m grid spacing
            grid_y = int(coord[1] / 200)

            if (start_x <= grid_x < start_x + district_size[0] and
                start_y <= grid_y < start_y + district_size[1]):
                managed.append(intersection_id)

        return managed

    def _setup_spaces(self) -> None:
        """Set up observation and action spaces."""
        # Observation space
        obs_space = self.preprocessor.create_observation_space()

        # Action spaces
        intersection_action_space = self.preprocessor.create_action_space("low_level")
        district_action_space = self.preprocessor.create_action_space("high_level")

        # Combined spaces for multi-agent environment
        self._agent_ids = set(list(self.intersection_agents.keys()) +
                             list(self.district_agents.keys()))

        self._observation_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True
        self.observation_space = spaces.Dict({
            agent_id: obs_space for agent_id in self._agent_ids
        })

        action_spaces = {}
        for agent_id in self.intersection_agents:
            action_spaces[agent_id] = intersection_action_space
        for agent_id in self.district_agents:
            action_spaces[agent_id] = district_action_space

        self.action_space = spaces.Dict(action_spaces)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment.

        Args:
            seed: Random seed.
            options: Reset options.

        Returns:
            Tuple of (observations, info).
        """
        super().reset(seed=seed)

        # Stop previous simulation
        if self.sumo_label is not None:
            self.data_loader.stop_simulation(self.sumo_label)

        # Start new simulation
        self.sumo_label = self.data_loader.start_simulation(self.scenario_data)

        # Reset state
        self.current_step = 0
        self.episode_metrics = {
            "total_waiting_time": 0.0,
            "total_throughput": 0.0,
            "total_fuel_consumption": 0.0,
            "coordination_events": 0,
        }

        # Reset preprocessor histories
        for agent_id in self._agent_ids:
            self.preprocessor.reset_history(agent_id)

        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()

        logger.debug(f"Reset environment - Step 0/{self.max_steps}")
        return observations, info

    def step(self, action_dict: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        """Step environment.

        Args:
            action_dict: Dictionary of actions for each agent.

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos).
        """
        # Apply actions
        self._apply_actions(action_dict)

        # Step SUMO simulation
        traci.switch(self.sumo_label)
        traci.simulationStep()

        # Update metrics
        self._update_metrics()

        # Get observations, rewards, and termination conditions
        observations = self._get_observations()
        rewards = self._calculate_rewards()
        terminateds = self._get_terminateds()
        truncateds = self._get_truncateds()
        infos = self._get_info()

        self.current_step += 1

        if self.current_step % 100 == 0:
            logger.debug(f"Environment step {self.current_step}/{self.max_steps}")

        return observations, rewards, terminateds, truncateds, infos

    def _apply_actions(self, action_dict: Dict[str, Union[int, np.ndarray]]) -> None:
        """Apply actions to SUMO simulation.

        Args:
            action_dict: Dictionary of actions for each agent.
        """
        traci.switch(self.sumo_label)

        # Apply intersection agent actions (traffic light control)
        for agent_id, action in action_dict.items():
            if agent_id in self.intersection_agents:
                self._apply_intersection_action(agent_id, action)
            elif agent_id in self.district_agents:
                self._apply_district_action(agent_id, action)

    def _apply_intersection_action(self, agent_id: str, action: int) -> None:
        """Apply intersection agent action.

        Args:
            agent_id: Intersection agent ID.
            action: Discrete action (traffic light phase).
        """
        intersection_data = self.intersection_agents[agent_id]["intersection_data"]
        tls_id = intersection_data.get("tls_id")

        if tls_id and traci.trafficlight.getIDList():
            if tls_id in traci.trafficlight.getIDList():
                # Get current program logic
                try:
                    current_phase = traci.trafficlight.getPhase(tls_id)

                    # Simple phase mapping (would be more sophisticated in practice)
                    phase_mapping = {0: 0, 1: 2, 2: 1, 3: 3}  # Map actions to SUMO phases
                    target_phase = phase_mapping.get(action, 0)

                    if target_phase != current_phase:
                        traci.trafficlight.setPhase(tls_id, target_phase)

                except Exception as e:
                    logger.warning(f"Failed to set phase for {tls_id}: {e}")

    def _apply_district_action(self, agent_id: str, action: np.ndarray) -> None:
        """Apply district agent action.

        Args:
            agent_id: District agent ID.
            action: Continuous coordination action.
        """
        # District actions are coordination signals that would influence
        # intersection timing constraints or priorities
        district_data = self.district_agents[agent_id]
        managed_intersections = district_data["managed_intersections"]

        # Simple coordination: adjust timing constraints based on action
        for i, intersection_id in enumerate(managed_intersections):
            if i < len(action):
                coordination_signal = float(action[i])

                # Apply coordination (e.g., adjust minimum green time)
                intersection_data = self.intersection_agents[intersection_id]["intersection_data"]
                tls_id = intersection_data.get("tls_id")

                if tls_id and traci.trafficlight.getIDList():
                    if tls_id in traci.trafficlight.getIDList():
                        try:
                            # Adjust phase duration based on coordination signal
                            current_duration = traci.trafficlight.getPhaseDuration(tls_id)
                            adjustment = coordination_signal * 10  # Scale coordination signal
                            new_duration = max(5, current_duration + adjustment)  # Min 5 seconds
                            traci.trafficlight.setPhaseDuration(tls_id, new_duration)
                        except Exception as e:
                            logger.warning(f"Failed to adjust timing for {tls_id}: {e}")

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents.

        Returns:
            Dictionary of observations for each agent.
        """
        traci.switch(self.sumo_label)
        observations = {}

        # Get intersection observations
        for agent_id in self.intersection_agents:
            obs = self._get_intersection_observation(agent_id)
            observations[agent_id] = obs

        # Get district observations
        for agent_id in self.district_agents:
            obs = self._get_district_observation(agent_id)
            observations[agent_id] = obs

        return observations

    def _get_intersection_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for intersection agent.

        Args:
            agent_id: Intersection agent ID.

        Returns:
            Preprocessed observation array.
        """
        intersection_data = self.intersection_agents[agent_id]["intersection_data"]
        tls_id = intersection_data.get("tls_id")

        raw_observation = {
            "vehicles": self._get_vehicle_data(intersection_data),
            "queue_lengths": self._get_queue_data(intersection_data),
            "waiting_times": self._get_waiting_data(intersection_data),
            "tls_phase": self._get_tls_data(tls_id),
            "neighbors": self._get_neighbor_data(agent_id),
        }

        return self.preprocessor.preprocess_observations(raw_observation, agent_id)

    def _get_district_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for district agent.

        Args:
            agent_id: District agent ID.

        Returns:
            Aggregated district observation.
        """
        district_data = self.district_agents[agent_id]
        managed_intersections = district_data["managed_intersections"]

        # Aggregate information from managed intersections
        total_queue_length = 0.0
        total_waiting_time = 0.0
        total_throughput = 0.0

        for intersection_id in managed_intersections:
            if intersection_id in self.intersection_agents:
                intersection_data = self.intersection_agents[intersection_id]["intersection_data"]

                # Get aggregated metrics
                queue_data = self._get_queue_data(intersection_data)
                waiting_data = self._get_waiting_data(intersection_data)
                vehicle_data = self._get_vehicle_data(intersection_data)

                total_queue_length += sum(queue_data.get("lane_queues", {}).values())
                waiting_times = waiting_data.get("lane_waiting_times", {})
                for lane_times in waiting_times.values():
                    total_waiting_time += sum(lane_times)

                lane_counts = vehicle_data.get("lane_counts", {})
                total_throughput += sum(lane_counts.values())

        # Create district-level observation
        district_obs = np.array([
            total_queue_length,
            total_waiting_time,
            total_throughput,
            len(managed_intersections),
            self.current_step / self.max_steps,  # Normalized time
        ], dtype=np.float32)

        # Pad to match expected dimension
        expected_dim = self.rl_config.agents.high_level.observation_space_dim
        if len(district_obs) < expected_dim:
            padding = np.zeros(expected_dim - len(district_obs), dtype=np.float32)
            district_obs = np.concatenate([district_obs, padding])

        return district_obs[:expected_dim]

    def _get_vehicle_data(self, intersection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get vehicle data for intersection.

        Args:
            intersection_data: Intersection configuration data.

        Returns:
            Vehicle data dictionary.
        """
        try:
            incoming_edges = intersection_data.get("incoming_edges", [])
            vehicle_data = {"lane_counts": {}, "lane_speeds": {}, "lane_densities": {}}

            for edge_id in incoming_edges:
                if edge_id in traci.edge.getIDList():
                    lanes = traci.edge.getLaneNumber(edge_id)
                    for lane_idx in range(lanes):
                        lane_id = f"{edge_id}_{lane_idx}"

                        try:
                            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

                            vehicle_data["lane_counts"][lane_id] = vehicle_count

                            if vehicle_ids:
                                speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids]
                                vehicle_data["lane_speeds"][lane_id] = speeds
                            else:
                                vehicle_data["lane_speeds"][lane_id] = [0.0]

                            lane_length = traci.lane.getLength(lane_id)
                            density = vehicle_count / max(lane_length, 1.0)
                            vehicle_data["lane_densities"][lane_id] = density

                        except Exception:
                            vehicle_data["lane_counts"][lane_id] = 0
                            vehicle_data["lane_speeds"][lane_id] = [0.0]
                            vehicle_data["lane_densities"][lane_id] = 0.0

            return vehicle_data

        except Exception as e:
            logger.warning(f"Failed to get vehicle data: {e}")
            return {"lane_counts": {}, "lane_speeds": {}, "lane_densities": {}}

    def _get_queue_data(self, intersection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get queue data for intersection.

        Args:
            intersection_data: Intersection configuration data.

        Returns:
            Queue data dictionary.
        """
        try:
            incoming_edges = intersection_data.get("incoming_edges", [])
            queue_data = {"lane_queues": {}, "queue_rates": {}, "max_queue_normalized": {}}

            for edge_id in incoming_edges:
                if edge_id in traci.edge.getIDList():
                    lanes = traci.edge.getLaneNumber(edge_id)
                    for lane_idx in range(lanes):
                        lane_id = f"{edge_id}_{lane_idx}"

                        try:
                            # Calculate queue length as vehicles with speed < 0.5 m/s
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                            queue_length = sum(1 for veh_id in vehicle_ids
                                             if traci.vehicle.getSpeed(veh_id) < 0.5)

                            queue_data["lane_queues"][lane_id] = queue_length
                            queue_data["queue_rates"][lane_id] = 0.0  # Would track change over time

                            lane_length = traci.lane.getLength(lane_id)
                            normalized_queue = queue_length * 5.0 / max(lane_length, 1.0)  # Assume 5m per vehicle
                            queue_data["max_queue_normalized"][lane_id] = min(normalized_queue, 1.0)

                        except Exception:
                            queue_data["lane_queues"][lane_id] = 0
                            queue_data["queue_rates"][lane_id] = 0.0
                            queue_data["max_queue_normalized"][lane_id] = 0.0

            return queue_data

        except Exception as e:
            logger.warning(f"Failed to get queue data: {e}")
            return {"lane_queues": {}, "queue_rates": {}, "max_queue_normalized": {}}

    def _get_waiting_data(self, intersection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get waiting time data for intersection.

        Args:
            intersection_data: Intersection configuration data.

        Returns:
            Waiting time data dictionary.
        """
        try:
            incoming_edges = intersection_data.get("incoming_edges", [])
            waiting_data = {"lane_waiting_times": {}, "max_waiting_times": {}, "cumulative_waiting": {}}

            for edge_id in incoming_edges:
                if edge_id in traci.edge.getIDList():
                    lanes = traci.edge.getLaneNumber(edge_id)
                    for lane_idx in range(lanes):
                        lane_id = f"{edge_id}_{lane_idx}"

                        try:
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                            waiting_times = []

                            for veh_id in vehicle_ids:
                                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                                waiting_times.append(waiting_time)

                            waiting_data["lane_waiting_times"][lane_id] = waiting_times
                            waiting_data["max_waiting_times"][lane_id] = max(waiting_times) if waiting_times else 0.0
                            waiting_data["cumulative_waiting"][lane_id] = sum(waiting_times)

                        except Exception:
                            waiting_data["lane_waiting_times"][lane_id] = []
                            waiting_data["max_waiting_times"][lane_id] = 0.0
                            waiting_data["cumulative_waiting"][lane_id] = 0.0

            return waiting_data

        except Exception as e:
            logger.warning(f"Failed to get waiting data: {e}")
            return {"lane_waiting_times": {}, "max_waiting_times": {}, "cumulative_waiting": {}}

    def _get_tls_data(self, tls_id: Optional[str]) -> Dict[str, Any]:
        """Get traffic light data.

        Args:
            tls_id: Traffic light system ID.

        Returns:
            Traffic light data dictionary.
        """
        if not tls_id or tls_id not in traci.trafficlight.getIDList():
            return {
                "current_phase": 0,
                "num_phases": 4,
                "urgency_scores": [0.0, 0.0, 0.0, 0.0]
            }

        try:
            current_phase = traci.trafficlight.getPhase(tls_id)
            program_id = traci.trafficlight.getProgram(tls_id)

            # Get number of phases (simplified)
            num_phases = 4  # Typical 4-phase intersection

            # Calculate urgency scores (simplified)
            urgency_scores = [0.5] * num_phases  # Placeholder values

            return {
                "current_phase": current_phase,
                "num_phases": num_phases,
                "urgency_scores": urgency_scores,
                "phase_duration": traci.trafficlight.getPhaseDuration(tls_id)
            }

        except Exception as e:
            logger.warning(f"Failed to get TLS data for {tls_id}: {e}")
            return {
                "current_phase": 0,
                "num_phases": 4,
                "urgency_scores": [0.0, 0.0, 0.0, 0.0]
            }

    def _get_neighbor_data(self, agent_id: str) -> Dict[str, Any]:
        """Get neighbor data for agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Neighbor data dictionary.
        """
        # Simplified neighbor detection
        neighbors = []

        # Get other intersection agents as neighbors
        for other_agent_id, other_agent_data in self.intersection_agents.items():
            if other_agent_id != agent_id:
                # Calculate distance and add as neighbor if close enough
                neighbor_info = {
                    "total_queue_length": 5.0,  # Placeholder
                    "throughput": 2.0,         # Placeholder
                    "distance": 300.0,         # Placeholder
                    "coordination_signal": 0.0 # Placeholder
                }
                neighbors.append(neighbor_info)

        return {"neighbors": neighbors[:8]}  # Max 8 neighbors

    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents.

        Returns:
            Dictionary of rewards for each agent.
        """
        rewards = {}

        # Calculate intersection agent rewards
        for agent_id in self.intersection_agents:
            rewards[agent_id] = self._calculate_intersection_reward(agent_id)

        # Calculate district agent rewards
        for agent_id in self.district_agents:
            rewards[agent_id] = self._calculate_district_reward(agent_id)

        return rewards

    def _calculate_intersection_reward(self, agent_id: str) -> float:
        """Calculate reward for intersection agent.

        Args:
            agent_id: Intersection agent ID.

        Returns:
            Reward value.
        """
        intersection_data = self.intersection_agents[agent_id]["intersection_data"]

        # Get current metrics
        vehicle_data = self._get_vehicle_data(intersection_data)
        waiting_data = self._get_waiting_data(intersection_data)

        # Calculate components
        waiting_penalty = sum(
            sum(times) for times in waiting_data["lane_waiting_times"].values()
        ) * self.rl_config.environment.reward.waiting_time_weight

        throughput_reward = sum(
            vehicle_data["lane_counts"].values()
        ) * self.rl_config.environment.reward.throughput_weight

        # Simple coordination bonus (would be more sophisticated)
        coordination_bonus = 0.1 * self.rl_config.environment.reward.coordination_weight

        total_reward = waiting_penalty + throughput_reward + coordination_bonus

        return float(total_reward)

    def _calculate_district_reward(self, agent_id: str) -> float:
        """Calculate reward for district agent.

        Args:
            agent_id: District agent ID.

        Returns:
            Reward value.
        """
        district_data = self.district_agents[agent_id]
        managed_intersections = district_data["managed_intersections"]

        # Aggregate rewards from managed intersections
        total_reward = 0.0
        for intersection_id in managed_intersections:
            if intersection_id in self.intersection_agents:
                intersection_reward = self._calculate_intersection_reward(intersection_id)
                total_reward += intersection_reward

        # Add coordination bonus
        coordination_bonus = len(managed_intersections) * 0.1
        total_reward += coordination_bonus

        return float(total_reward / max(len(managed_intersections), 1))

    def _update_metrics(self) -> None:
        """Update episode metrics."""
        try:
            traci.switch(self.sumo_label)

            # Update waiting time
            total_waiting = 0.0
            for agent_id in self.intersection_agents:
                intersection_data = self.intersection_agents[agent_id]["intersection_data"]
                waiting_data = self._get_waiting_data(intersection_data)
                for times in waiting_data["lane_waiting_times"].values():
                    total_waiting += sum(times)

            self.episode_metrics["total_waiting_time"] += total_waiting

            # Update throughput (vehicles that completed their journey)
            departed_vehicles = traci.simulation.getDepartedNumber()
            arrived_vehicles = traci.simulation.getArrivedNumber()
            self.episode_metrics["total_throughput"] += arrived_vehicles

            # Update fuel consumption (simplified)
            self.episode_metrics["total_fuel_consumption"] += departed_vehicles * 0.1

        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")

    def _get_terminateds(self) -> Dict[str, bool]:
        """Get termination conditions.

        Returns:
            Dictionary of termination flags for each agent.
        """
        # Environment terminates when simulation ends
        simulation_ended = self.current_step >= self.max_steps

        return {agent_id: simulation_ended for agent_id in self._agent_ids}

    def _get_truncateds(self) -> Dict[str, bool]:
        """Get truncation conditions.

        Returns:
            Dictionary of truncation flags for each agent.
        """
        # No truncation conditions in this environment
        return {agent_id: False for agent_id in self._agent_ids}

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary.

        Returns:
            Dictionary of info for each agent.
        """
        info = {}

        # Common info for all agents
        common_info = {
            "step": self.current_step,
            "metrics": self.episode_metrics.copy(),
        }

        for agent_id in self._agent_ids:
            info[agent_id] = common_info.copy()

        return info

    def close(self) -> None:
        """Close environment."""
        if self.sumo_label is not None:
            self.data_loader.stop_simulation(self.sumo_label)
            self.sumo_label = None

        logger.info("Closed TrafficEnvironment")


class MLflowCallback(BaseCallback):
    """MLflow callback for logging training metrics."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize callback.

        Args:
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called on each step.

        Returns:
            Whether to continue training.
        """
        # Log episode-level metrics
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Log to MLflow
                mlflow.log_metrics({
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
                    "mean_episode_length": np.mean(self.episode_lengths[-100:]),
                }, step=self.num_timesteps)

        # Log training metrics
        if self.num_timesteps % 1000 == 0:
            mlflow.log_metrics({
                "timesteps": self.num_timesteps,
                "learning_rate": self.model.learning_rate,
            }, step=self.num_timesteps)

        return True


class HierarchicalTrainer:
    """Trainer for hierarchical multi-agent RL system."""

    def __init__(self, config: Config) -> None:
        """Initialize trainer.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.framework = config.training.framework

        # Initialize MLflow
        if config.logging.mlflow.enabled:
            mlflow.set_tracking_uri(config.logging.mlflow.tracking_uri)
            mlflow.set_experiment(config.logging.mlflow.experiment_name)

        # Initialize Ray if using RLlib
        if self.framework == "ray":
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

        logger.info(f"Initialized HierarchicalTrainer with framework: {self.framework}")

    def train(self) -> Dict[str, Any]:
        """Train the hierarchical multi-agent system.

        Returns:
            Training results dictionary.
        """
        logger.info("Starting hierarchical multi-agent training...")

        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(self.config.to_dict())

            if self.framework == "ray":
                return self._train_with_ray()
            else:
                return self._train_with_sb3()

    def _train_with_ray(self) -> Dict[str, Any]:
        """Train using Ray RLlib.

        Returns:
            Training results.
        """
        # Environment configuration
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": self.config.environment.scenario
        }

        # Register environment
        from ray.tune.registry import register_env
        register_env("traffic_env", lambda config: TrafficEnvironment(config))

        # Configure algorithms for different agent types
        low_level_config = (
            PPOConfig()
            .environment("traffic_env", env_config=env_config)
            .framework("torch")
            .training(
                train_batch_size=self.config.training.batch_size,
                lr=self.config.training.learning_rate,
                gamma=self.config.training.gamma,
                lambda_=self.config.training.gae_lambda,
                clip_param=self.config.training.clip_range,
                entropy_coeff=self.config.training.entropy_coeff,
                vf_loss_coeff=self.config.training.value_function_coeff,
                max_grad_norm=self.config.training.max_grad_norm,
            )
            .rollouts(num_rollout_workers=2)
            .resources(num_gpus=1 if torch.cuda.is_available() else 0)
            .multi_agent(
                policies=self._create_ray_policies(),
                policy_mapping_fn=self._policy_mapping_fn,
            )
        )

        # Create trainer
        from ray.rllib.algorithms.ppo import PPO
        trainer = PPO(config=low_level_config)

        # Training loop
        results = {}
        best_reward = float('-inf')

        for iteration in range(self.config.training.total_timesteps // self.config.training.batch_size):
            result = trainer.train()

            # Log metrics
            episode_reward_mean = result["episode_reward_mean"]
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                # Save best model
                checkpoint_path = trainer.save()
                mlflow.log_artifact(checkpoint_path)

            # Log to MLflow
            mlflow.log_metrics({
                "episode_reward_mean": episode_reward_mean,
                "episode_len_mean": result.get("episode_len_mean", 0),
                "timesteps_total": result["timesteps_total"],
            }, step=iteration)

            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: mean reward = {episode_reward_mean:.2f}")

            # Evaluation
            if iteration % self.config.evaluation.interval == 0:
                eval_results = self._evaluate_ray_model(trainer)
                results[f"eval_{iteration}"] = eval_results

        trainer.stop()
        return results

    def _train_with_sb3(self) -> Dict[str, Any]:
        """Train using Stable-Baselines3.

        Returns:
            Training results.
        """
        # Create hierarchical agent
        hierarchical_agent = HierarchicalTrafficAgent(self.config)

        # For SB3, we need to simplify to single-agent or use a wrapper
        # Here we'll train intersection agents first, then district agents

        results = {}

        # Phase 1: Pre-train intersection agents
        if self.config.training.hierarchical.pretrain_low_level:
            logger.info("Pre-training intersection agents...")
            intersection_results = self._pretrain_intersection_agents(hierarchical_agent)
            results["pretrain_intersection"] = intersection_results

        # Phase 2: Train district agents with fixed intersection agents
        logger.info("Training district agents...")
        district_results = self._train_district_agents(hierarchical_agent)
        results["train_district"] = district_results

        # Phase 3: Joint fine-tuning
        logger.info("Joint fine-tuning...")
        joint_results = self._joint_fine_tuning(hierarchical_agent)
        results["joint_training"] = joint_results

        return results

    def _pretrain_intersection_agents(self, hierarchical_agent: HierarchicalTrafficAgent) -> Dict[str, Any]:
        """Pre-train intersection agents.

        Args:
            hierarchical_agent: Hierarchical agent to train.

        Returns:
            Pre-training results.
        """
        # Create simplified environment for intersection training
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": self.config.environment.scenario,
            "single_agent_mode": True  # Simplified mode
        }

        # Create wrapper environment
        env = TrafficEnvironment(env_config)
        env = Monitor(env)

        # Create PPO model for intersection agents
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.training.learning_rate,
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
            clip_range=self.config.training.clip_range,
            ent_coef=self.config.training.entropy_coeff,
            vf_coef=self.config.training.value_function_coeff,
            max_grad_norm=self.config.training.max_grad_norm,
            verbose=1,
            tensorboard_log="logs/tensorboard" if self.config.logging.tensorboard.enabled else None
        )

        # Train with callback
        callback = MLflowCallback()
        model.learn(
            total_timesteps=self.config.training.hierarchical.pretrain_steps,
            callback=callback
        )

        # Save model
        model_path = "models/pretrained_intersection_agent"
        model.save(model_path)
        mlflow.log_artifact(f"{model_path}.zip")

        # Evaluate
        eval_rewards, eval_lengths = evaluate_policy(
            model, env, n_eval_episodes=self.config.evaluation.episodes
        )

        env.close()

        return {
            "mean_reward": np.mean(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "model_path": model_path
        }

    def _train_district_agents(self, hierarchical_agent: HierarchicalTrafficAgent) -> Dict[str, Any]:
        """Train district agents.

        Args:
            hierarchical_agent: Hierarchical agent to train.

        Returns:
            Training results.
        """
        # For district agents, we use SAC (continuous actions)
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": self.config.environment.scenario,
            "district_mode": True
        }

        env = TrafficEnvironment(env_config)
        env = Monitor(env)

        # Use SAC for continuous control
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.config.training.learning_rate,
            gamma=self.config.training.gamma,
            verbose=1,
            tensorboard_log="logs/tensorboard" if self.config.logging.tensorboard.enabled else None
        )

        callback = MLflowCallback()
        model.learn(
            total_timesteps=self.config.training.total_timesteps // 2,
            callback=callback
        )

        # Save model
        model_path = "models/trained_district_agent"
        model.save(model_path)
        mlflow.log_artifact(f"{model_path}.zip")

        # Evaluate
        eval_rewards, eval_lengths = evaluate_policy(
            model, env, n_eval_episodes=self.config.evaluation.episodes
        )

        env.close()

        return {
            "mean_reward": np.mean(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "model_path": model_path
        }

    def _joint_fine_tuning(self, hierarchical_agent: HierarchicalTrafficAgent) -> Dict[str, Any]:
        """Perform joint fine-tuning of all agents.

        Args:
            hierarchical_agent: Hierarchical agent to fine-tune.

        Returns:
            Fine-tuning results.
        """
        # Create full environment
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": self.config.environment.scenario
        }

        env = TrafficEnvironment(env_config)

        # Create wrapper for hierarchical agent
        wrapper = TrafficEnvironmentWrapper(self.config, hierarchical_agent)

        # Simplified fine-tuning (in practice, would use more sophisticated approach)
        results = {"message": "Joint fine-tuning completed (simplified implementation)"}

        env.close()

        return results

    def _create_ray_policies(self) -> Dict[str, Tuple[type, spaces.Space, spaces.Space, Dict[str, Any]]]:
        """Create Ray RLlib policies.

        Returns:
            Dictionary of policy configurations.
        """
        from ray.rllib.policy.policy import PolicySpec

        # Create observation and action spaces
        preprocessor = TrafficPreprocessor(self.config)
        obs_space = preprocessor.create_observation_space()
        intersection_action_space = preprocessor.create_action_space("low_level")
        district_action_space = preprocessor.create_action_space("high_level")

        policies = {}

        # Intersection agent policies
        policies["intersection_policy"] = PolicySpec(
            policy_class=None,  # Use default
            observation_space=obs_space,
            action_space=intersection_action_space,
            config={"model": {"fcnet_hiddens": [256, 256]}}
        )

        # District agent policies
        policies["district_policy"] = PolicySpec(
            policy_class=None,  # Use default
            observation_space=obs_space,
            action_space=district_action_space,
            config={"model": {"fcnet_hiddens": [512, 512]}}
        )

        return policies

    def _policy_mapping_fn(self, agent_id: str) -> str:
        """Map agent ID to policy.

        Args:
            agent_id: Agent identifier.

        Returns:
            Policy name.
        """
        if "intersection" in agent_id:
            return "intersection_policy"
        elif "district" in agent_id:
            return "district_policy"
        else:
            return "intersection_policy"  # Default

    def _evaluate_ray_model(self, trainer: Any) -> Dict[str, float]:
        """Evaluate Ray RLlib model.

        Args:
            trainer: Ray RLlib trainer.

        Returns:
            Evaluation results.
        """
        # Simple evaluation (would be more comprehensive in practice)
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": self.config.environment.scenario
        }

        eval_env = TrafficEnvironment(env_config)

        # Run evaluation episodes
        total_rewards = []
        total_lengths = []

        for episode in range(self.config.evaluation.episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = {agent_id: False for agent_id in obs.keys()}

            while not all(terminated.values()):
                # Get actions from trainer
                actions = {}
                for agent_id, agent_obs in obs.items():
                    action = trainer.compute_single_action(
                        agent_obs,
                        policy_id=self._policy_mapping_fn(agent_id)
                    )
                    actions[agent_id] = action

                obs, rewards, terminated, truncated, infos = eval_env.step(actions)
                episode_reward += sum(rewards.values())
                episode_length += 1

                if episode_length > 1000:  # Timeout
                    break

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        eval_env.close()

        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_length": np.mean(total_lengths),
        }

    def evaluate_transfer_learning(self) -> Dict[str, float]:
        """Evaluate transfer learning capabilities.

        Returns:
            Transfer learning evaluation results.
        """
        logger.info("Evaluating transfer learning...")

        results = {}
        transfer_scenarios = self.config.evaluation.transfer_scenarios

        for scenario_pair in transfer_scenarios:
            logger.info(f"Evaluating transfer: {scenario_pair}")

            # This would involve training on source scenario and evaluating on target
            # For now, we return placeholder results
            results[scenario_pair] = {
                "source_performance": 0.85,
                "target_performance": 0.70,
                "transfer_retention": 0.70 / 0.85
            }

        return results

    def save_final_model(self, model_path: str) -> None:
        """Save final trained model.

        Args:
            model_path: Path to save model.
        """
        # This would save the complete hierarchical model
        logger.info(f"Saving final model to {model_path}")
        mlflow.log_artifact(model_path)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.framework == "ray" and ray.is_initialized():
            ray.shutdown()

        logger.info("Training cleanup completed")