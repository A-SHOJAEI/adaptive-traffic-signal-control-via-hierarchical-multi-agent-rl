"""Data preprocessing utilities for traffic simulation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..utils.config import Config

logger = logging.getLogger(__name__)


class TrafficPreprocessor:
    """Preprocessor for traffic simulation data."""

    def __init__(self, config: Config) -> None:
        """Initialize traffic preprocessor.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.observation_history: Dict[str, List[np.ndarray]] = {}
        self.normalization_params: Dict[str, Dict[str, float]] = {}

    def preprocess_observations(
        self,
        raw_observations: Dict[str, any],
        agent_id: str
    ) -> np.ndarray:
        """Preprocess raw observations for an agent.

        Args:
            raw_observations: Raw observation data from simulation.
            agent_id: ID of the agent.

        Returns:
            Preprocessed observation vector.
        """
        # Extract relevant features
        features = []

        # Vehicle positions and counts
        if self.config.environment.observation.vehicle_positions:
            vehicle_features = self._extract_vehicle_features(
                raw_observations.get("vehicles", {})
            )
            features.extend(vehicle_features)

        # Queue lengths
        if self.config.environment.observation.queue_lengths:
            queue_features = self._extract_queue_features(
                raw_observations.get("queue_lengths", {})
            )
            features.extend(queue_features)

        # Waiting times
        if self.config.environment.observation.waiting_times:
            waiting_features = self._extract_waiting_features(
                raw_observations.get("waiting_times", {})
            )
            features.extend(waiting_features)

        # Traffic light phases
        if self.config.environment.observation.phase_durations:
            phase_features = self._extract_phase_features(
                raw_observations.get("tls_phase", {}),
                raw_observations.get("phase_duration", 0)
            )
            features.extend(phase_features)

        # Neighboring states
        if self.config.environment.observation.neighboring_states:
            neighbor_features = self._extract_neighbor_features(
                raw_observations.get("neighbors", {}),
                agent_id
            )
            features.extend(neighbor_features)

        # Convert to numpy array
        observation = np.array(features, dtype=np.float32)

        # Apply normalization
        observation = self._normalize_observation(observation, agent_id)

        # Add temporal information
        observation = self._add_temporal_features(observation, agent_id)

        return observation

    def _extract_vehicle_features(self, vehicle_data: Dict[str, any]) -> List[float]:
        """Extract features related to vehicle positions and counts.

        Args:
            vehicle_data: Vehicle-related data.

        Returns:
            List of vehicle features.
        """
        features = []

        # Vehicle counts per lane
        lane_counts = vehicle_data.get("lane_counts", {})
        for lane_id in sorted(lane_counts.keys()):
            features.append(float(lane_counts[lane_id]))

        # Average speeds per lane
        lane_speeds = vehicle_data.get("lane_speeds", {})
        for lane_id in sorted(lane_speeds.keys()):
            avg_speed = np.mean(lane_speeds[lane_id]) if lane_speeds[lane_id] else 0.0
            features.append(avg_speed)

        # Density indicators
        lane_densities = vehicle_data.get("lane_densities", {})
        for lane_id in sorted(lane_densities.keys()):
            features.append(float(lane_densities[lane_id]))

        # Fill missing lanes with zeros
        expected_lanes = 12  # Assume max 4 approaches * 3 lanes each
        while len(features) < expected_lanes * 3:  # counts + speeds + densities
            features.append(0.0)

        return features[:expected_lanes * 3]  # Truncate if too many

    def _extract_queue_features(self, queue_data: Dict[str, any]) -> List[float]:
        """Extract features related to queue lengths.

        Args:
            queue_data: Queue-related data.

        Returns:
            List of queue features.
        """
        features = []

        # Queue lengths per lane
        lane_queues = queue_data.get("lane_queues", {})
        for lane_id in sorted(lane_queues.keys()):
            features.append(float(lane_queues[lane_id]))

        # Queue growth rates
        queue_rates = queue_data.get("queue_rates", {})
        for lane_id in sorted(queue_rates.keys()):
            features.append(float(queue_rates[lane_id]))

        # Maximum queue lengths (normalized by lane length)
        max_queues = queue_data.get("max_queue_normalized", {})
        for lane_id in sorted(max_queues.keys()):
            features.append(float(max_queues[lane_id]))

        # Fill missing features
        expected_features = 12 * 3  # 12 lanes * 3 queue metrics
        while len(features) < expected_features:
            features.append(0.0)

        return features[:expected_features]

    def _extract_waiting_features(self, waiting_data: Dict[str, any]) -> List[float]:
        """Extract features related to waiting times.

        Args:
            waiting_data: Waiting time data.

        Returns:
            List of waiting time features.
        """
        features = []

        # Average waiting times per lane
        lane_waiting = waiting_data.get("lane_waiting_times", {})
        for lane_id in sorted(lane_waiting.keys()):
            avg_waiting = np.mean(lane_waiting[lane_id]) if lane_waiting[lane_id] else 0.0
            features.append(avg_waiting)

        # Maximum waiting times per lane
        max_waiting = waiting_data.get("max_waiting_times", {})
        for lane_id in sorted(max_waiting.keys()):
            features.append(float(max_waiting[lane_id]))

        # Cumulative waiting times
        cum_waiting = waiting_data.get("cumulative_waiting", {})
        for lane_id in sorted(cum_waiting.keys()):
            features.append(float(cum_waiting[lane_id]))

        # Fill missing features
        expected_features = 12 * 3  # 12 lanes * 3 waiting metrics
        while len(features) < expected_features:
            features.append(0.0)

        return features[:expected_features]

    def _extract_phase_features(
        self,
        phase_data: Dict[str, any],
        phase_duration: float
    ) -> List[float]:
        """Extract features related to traffic light phases.

        Args:
            phase_data: Traffic light phase data.
            phase_duration: Current phase duration.

        Returns:
            List of phase features.
        """
        features = []

        # Current phase (one-hot encoded)
        current_phase = phase_data.get("current_phase", 0)
        num_phases = phase_data.get("num_phases", 4)

        for i in range(num_phases):
            features.append(1.0 if i == current_phase else 0.0)

        # Phase duration (normalized)
        max_duration = self.config.environment.max_green_duration
        normalized_duration = min(phase_duration / max_duration, 1.0)
        features.append(normalized_duration)

        # Time until next yellow phase
        min_green = self.config.environment.min_green_duration
        time_until_yellow = max(0, min_green - phase_duration)
        normalized_time = time_until_yellow / min_green
        features.append(normalized_time)

        # Phase urgency (based on queue pressures)
        urgency_scores = phase_data.get("urgency_scores", [0.0] * num_phases)
        features.extend(urgency_scores[:num_phases])

        return features

    def _extract_neighbor_features(
        self,
        neighbor_data: Dict[str, any],
        agent_id: str
    ) -> List[float]:
        """Extract features from neighboring intersections.

        Args:
            neighbor_data: Data from neighboring agents.
            agent_id: Current agent ID.

        Returns:
            List of neighbor features.
        """
        features = []

        neighbors = neighbor_data.get("neighbors", [])
        max_neighbors = 8  # Maximum number of neighbors to consider

        for i in range(max_neighbors):
            if i < len(neighbors):
                neighbor = neighbors[i]
                # Neighbor queue length summary
                neighbor_queues = neighbor.get("total_queue_length", 0.0)
                features.append(neighbor_queues)

                # Neighbor throughput
                neighbor_throughput = neighbor.get("throughput", 0.0)
                features.append(neighbor_throughput)

                # Distance to neighbor (normalized)
                distance = neighbor.get("distance", float('inf'))
                normalized_distance = min(distance / 1000.0, 1.0)  # Normalize by 1km
                features.append(normalized_distance)

                # Coordination signal from neighbor
                coordination_signal = neighbor.get("coordination_signal", 0.0)
                features.append(coordination_signal)
            else:
                # Fill with zeros for missing neighbors
                features.extend([0.0, 0.0, 1.0, 0.0])  # No queue, no throughput, max distance, no signal

        return features

    def _normalize_observation(self, observation: np.ndarray, agent_id: str) -> np.ndarray:
        """Normalize observation using running statistics.

        Args:
            observation: Raw observation vector.
            agent_id: Agent identifier.

        Returns:
            Normalized observation vector.
        """
        if agent_id not in self.normalization_params:
            self.normalization_params[agent_id] = {
                "mean": np.zeros_like(observation),
                "std": np.ones_like(observation),
                "count": 0,
            }

        params = self.normalization_params[agent_id]

        # Update running statistics
        params["count"] += 1
        alpha = 1.0 / params["count"] if params["count"] < 1000 else 0.001

        params["mean"] = (1 - alpha) * params["mean"] + alpha * observation
        squared_diff = (observation - params["mean"]) ** 2
        params["std"] = np.sqrt(
            (1 - alpha) * params["std"] ** 2 + alpha * squared_diff
        )

        # Avoid division by zero
        params["std"] = np.maximum(params["std"], 1e-8)

        # Apply normalization
        normalized_obs = (observation - params["mean"]) / params["std"]

        # Clip to reasonable range
        normalized_obs = np.clip(normalized_obs, -5.0, 5.0)

        return normalized_obs

    def _add_temporal_features(self, observation: np.ndarray, agent_id: str) -> np.ndarray:
        """Add temporal features to observation.

        Args:
            observation: Current observation.
            agent_id: Agent identifier.

        Returns:
            Observation with temporal features.
        """
        history_length = self.config.environment.observation.history_length

        if agent_id not in self.observation_history:
            self.observation_history[agent_id] = []

        # Add current observation to history
        self.observation_history[agent_id].append(observation.copy())

        # Keep only recent history
        if len(self.observation_history[agent_id]) > history_length:
            self.observation_history[agent_id] = \
                self.observation_history[agent_id][-history_length:]

        # Create temporal features
        history = self.observation_history[agent_id]

        if len(history) < history_length:
            # Pad with zeros if not enough history
            padding_size = observation.shape[0] * (history_length - len(history))
            temporal_obs = np.concatenate([
                np.zeros(padding_size),
                np.concatenate(history)
            ])
        else:
            temporal_obs = np.concatenate(history[-history_length:])

        return temporal_obs.astype(np.float32)

    def create_observation_space(self) -> spaces.Box:
        """Create Gymnasium observation space.

        Returns:
            Gymnasium Box space for observations.
        """
        # Calculate observation dimensions
        base_features = 0

        if self.config.environment.observation.vehicle_positions:
            base_features += 12 * 3  # 12 lanes * 3 vehicle metrics

        if self.config.environment.observation.queue_lengths:
            base_features += 12 * 3  # 12 lanes * 3 queue metrics

        if self.config.environment.observation.waiting_times:
            base_features += 12 * 3  # 12 lanes * 3 waiting metrics

        if self.config.environment.observation.phase_durations:
            base_features += 4 + 2 + 4  # 4 phases + 2 timing + 4 urgency scores

        if self.config.environment.observation.neighboring_states:
            base_features += 8 * 4  # 8 neighbors * 4 features each

        # Multiply by history length for temporal features
        history_length = self.config.environment.observation.history_length
        total_features = base_features * history_length

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

    def create_action_space(self, agent_type: str = "low_level") -> Union[spaces.Discrete, spaces.Box]:
        """Create Gymnasium action space for agent.

        Args:
            agent_type: Type of agent ("low_level" or "high_level").

        Returns:
            Gymnasium action space.
        """
        if agent_type == "low_level":
            # Discrete action space for traffic light phases
            return spaces.Discrete(self.config.agents.low_level.action_space_size)
        else:
            # Continuous action space for high-level coordination
            action_size = self.config.agents.high_level.action_space_size
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_size,),
                dtype=np.float32
            )

    def reset_history(self, agent_id: str) -> None:
        """Reset observation history for an agent.

        Args:
            agent_id: Agent identifier.
        """
        if agent_id in self.observation_history:
            self.observation_history[agent_id] = []

    def get_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get normalization statistics.

        Returns:
            Dictionary of normalization parameters for each agent.
        """
        return self.normalization_params.copy()


class SyntheticTrafficGenerator:
    """Generator for synthetic traffic scenarios."""

    def __init__(self, config: Config) -> None:
        """Initialize synthetic traffic generator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.rng = np.random.RandomState(config.experiment.seed)

    def generate_traffic_demand(
        self,
        network_topology: nx.DiGraph,
        time_horizon: int,
        demand_pattern: str = "uniform"
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Generate synthetic traffic demand.

        Args:
            network_topology: Network topology graph.
            time_horizon: Total simulation time in seconds.
            demand_pattern: Type of demand pattern ("uniform", "peak_hour", "random").

        Returns:
            Dictionary mapping time intervals to origin-destination demands.
        """
        nodes = list(network_topology.nodes())
        edges = list(network_topology.edges())

        if len(nodes) < 2:
            logger.warning("Network has insufficient nodes for traffic generation")
            return {}

        # Time discretization (5-minute intervals)
        time_step = 300  # 5 minutes
        time_intervals = list(range(0, time_horizon, time_step))

        demand_data = {}

        for t in time_intervals:
            demands = []

            if demand_pattern == "uniform":
                demands = self._generate_uniform_demand(nodes, t)
            elif demand_pattern == "peak_hour":
                demands = self._generate_peak_hour_demand(nodes, t, time_horizon)
            elif demand_pattern == "random":
                demands = self._generate_random_demand(nodes, t)
            else:
                raise ValueError(f"Unknown demand pattern: {demand_pattern}")

            demand_data[t] = demands

        logger.info(f"Generated {demand_pattern} traffic demand for {len(time_intervals)} intervals")
        return demand_data

    def _generate_uniform_demand(
        self,
        nodes: List[str],
        time: int
    ) -> List[Tuple[str, str, float]]:
        """Generate uniform traffic demand.

        Args:
            nodes: List of network nodes.
            time: Current time.

        Returns:
            List of (origin, destination, flow_rate) tuples.
        """
        demands = []
        base_flow_rate = 0.1  # vehicles per second

        for origin in nodes:
            for destination in nodes:
                if origin != destination:
                    flow_rate = base_flow_rate * self.rng.uniform(0.5, 2.0)
                    demands.append((origin, destination, flow_rate))

        return demands

    def _generate_peak_hour_demand(
        self,
        nodes: List[str],
        time: int,
        total_time: int
    ) -> List[Tuple[str, str, float]]:
        """Generate peak hour traffic demand pattern.

        Args:
            nodes: List of network nodes.
            time: Current time.
            total_time: Total simulation time.

        Returns:
            List of (origin, destination, flow_rate) tuples.
        """
        demands = []

        # Define peak hours (7-9 AM and 5-7 PM in simulation time)
        peak_morning = (0.2 * total_time, 0.35 * total_time)
        peak_evening = (0.65 * total_time, 0.8 * total_time)

        # Calculate demand multiplier based on time
        if peak_morning[0] <= time <= peak_morning[1]:
            multiplier = 2.5  # High demand during morning peak
        elif peak_evening[0] <= time <= peak_evening[1]:
            multiplier = 2.5  # High demand during evening peak
        elif abs(time - (peak_morning[0] + peak_morning[1]) / 2) < 0.1 * total_time:
            multiplier = 1.5  # Medium demand near peaks
        elif abs(time - (peak_evening[0] + peak_evening[1]) / 2) < 0.1 * total_time:
            multiplier = 1.5  # Medium demand near peaks
        else:
            multiplier = 0.5  # Low demand during off-peak

        base_flow_rate = 0.1 * multiplier

        for origin in nodes:
            for destination in nodes:
                if origin != destination:
                    # Add directional bias for peak hours
                    if peak_morning[0] <= time <= peak_morning[1]:
                        # Morning: more traffic from residential to business areas
                        direction_bias = 1.5 if origin < destination else 0.7
                    elif peak_evening[0] <= time <= peak_evening[1]:
                        # Evening: more traffic from business to residential areas
                        direction_bias = 0.7 if origin < destination else 1.5
                    else:
                        direction_bias = 1.0

                    flow_rate = base_flow_rate * direction_bias * self.rng.uniform(0.5, 1.5)
                    demands.append((origin, destination, flow_rate))

        return demands

    def _generate_random_demand(
        self,
        nodes: List[str],
        time: int
    ) -> List[Tuple[str, str, float]]:
        """Generate random traffic demand.

        Args:
            nodes: List of network nodes.
            time: Current time.

        Returns:
            List of (origin, destination, flow_rate) tuples.
        """
        demands = []
        num_od_pairs = self.rng.randint(len(nodes), len(nodes) ** 2)

        for _ in range(num_od_pairs):
            origin, destination = self.rng.choice(nodes, 2, replace=False)
            flow_rate = self.rng.exponential(scale=0.2)  # Exponential distribution
            demands.append((origin, destination, flow_rate))

        return demands

    def generate_incident_scenarios(
        self,
        network_topology: nx.DiGraph,
        time_horizon: int,
        incident_probability: float = 0.01
    ) -> List[Dict[str, any]]:
        """Generate random incident scenarios.

        Args:
            network_topology: Network topology graph.
            time_horizon: Total simulation time.
            incident_probability: Probability of incident per time step per edge.

        Returns:
            List of incident dictionaries.
        """
        incidents = []
        edges = list(network_topology.edges())

        time_step = 60  # Check for incidents every minute
        for t in range(0, time_horizon, time_step):
            for edge in edges:
                if self.rng.random() < incident_probability:
                    incident = {
                        "edge": edge,
                        "start_time": t,
                        "duration": self.rng.randint(300, 1800),  # 5-30 minutes
                        "severity": self.rng.choice(["minor", "major"], p=[0.7, 0.3]),
                        "lane_closure": self.rng.randint(1, 3),  # Number of lanes closed
                    }
                    incidents.append(incident)

        logger.info(f"Generated {len(incidents)} incident scenarios")
        return incidents

    def create_scenario_variations(
        self,
        base_scenario: Dict[str, any],
        num_variations: int = 10
    ) -> List[Dict[str, any]]:
        """Create variations of a base scenario.

        Args:
            base_scenario: Base scenario configuration.
            num_variations: Number of variations to create.

        Returns:
            List of scenario variations.
        """
        variations = []

        for i in range(num_variations):
            variation = base_scenario.copy()

            # Vary traffic demand
            demand_multiplier = self.rng.uniform(0.7, 1.3)
            variation["demand_multiplier"] = demand_multiplier

            # Vary incident probability
            incident_multiplier = self.rng.uniform(0.5, 2.0)
            variation["incident_multiplier"] = incident_multiplier

            # Vary signal timing constraints
            variation["min_green_multiplier"] = self.rng.uniform(0.8, 1.2)
            variation["max_green_multiplier"] = self.rng.uniform(0.8, 1.2)

            # Add random seed for this variation
            variation["seed"] = self.config.experiment.seed + i

            variations.append(variation)

        logger.info(f"Created {num_variations} scenario variations")
        return variations