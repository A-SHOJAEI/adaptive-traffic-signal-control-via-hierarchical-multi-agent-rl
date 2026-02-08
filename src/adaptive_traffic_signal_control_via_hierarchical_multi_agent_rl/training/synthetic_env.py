"""Synthetic traffic environment for training without SUMO dependency.

This module provides a lightweight, pure-Python traffic grid simulation that
mimics the interface of the SUMO-based TrafficEnvironment but uses a simplified
model of traffic dynamics. It supports multi-agent hierarchical RL training
with both intersection-level and district-level agents.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class SyntheticTrafficEnvironment(gym.Env):
    """A synthetic grid-world traffic environment for hierarchical multi-agent RL.

    This environment simulates a grid of intersections where each intersection
    has traffic lights controlling vehicle flow. Vehicles are modeled as
    aggregate flow quantities on each lane, rather than individual agents.

    The environment supports:
    - Multiple intersection agents (low-level, discrete actions)
    - District coordination agents (high-level, continuous actions)
    - Configurable grid sizes and traffic patterns
    - Reward signals based on waiting times, throughput, and coordination
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize synthetic traffic environment.

        Args:
            config: Environment configuration dictionary with keys:
                - grid_rows: Number of rows in the grid.
                - grid_cols: Number of columns in the grid.
                - max_steps: Maximum simulation steps per episode.
                - num_phases: Number of traffic light phases per intersection.
                - num_lanes_per_approach: Number of lanes per approach direction.
                - base_arrival_rate: Base vehicle arrival rate.
                - obs_dim: Observation dimension per agent.
                - high_level_obs_dim: Observation dimension for district agents.
                - high_level_action_dim: Action dimension for district agents.
                - district_size: Tuple of (rows, cols) per district.
                - coordination_frequency: How often high-level agents act.
                - seed: Random seed.
        """
        super().__init__()
        self.grid_rows = config.get("grid_rows", 5)
        self.grid_cols = config.get("grid_cols", 5)
        self.max_steps = config.get("max_steps", 500)
        self.num_phases = config.get("num_phases", 4)
        self.num_lanes = config.get("num_lanes_per_approach", 3)
        self.base_arrival_rate = config.get("base_arrival_rate", 0.3)
        self.obs_dim = config.get("obs_dim", 64)
        self.high_level_obs_dim = config.get("high_level_obs_dim", 128)
        self.high_level_action_dim = config.get("high_level_action_dim", 16)
        self.district_size = config.get("district_size", (3, 3))
        self.coordination_frequency = config.get("coordination_frequency", 10)
        self._seed = config.get("seed", 42)

        self.num_intersections = self.grid_rows * self.grid_cols

        # District layout
        self.num_districts_x = max(1, self.grid_cols // self.district_size[1])
        self.num_districts_y = max(1, self.grid_rows // self.district_size[0])
        self.num_districts = self.num_districts_x * self.num_districts_y

        # Agent IDs
        self.intersection_ids = [
            f"intersection_{r}_{c}"
            for r in range(self.grid_rows)
            for c in range(self.grid_cols)
        ]
        self.district_ids = [
            f"district_{dr}_{dc}"
            for dr in range(self.num_districts_y)
            for dc in range(self.num_districts_x)
        ]
        self.all_agent_ids = self.intersection_ids + self.district_ids

        # District-to-intersection mapping
        self.district_to_intersections: Dict[str, List[str]] = {}
        for dr in range(self.num_districts_y):
            for dc in range(self.num_districts_x):
                did = f"district_{dr}_{dc}"
                managed = []
                for r in range(dr * self.district_size[0],
                               min((dr + 1) * self.district_size[0], self.grid_rows)):
                    for c in range(dc * self.district_size[1],
                                   min((dc + 1) * self.district_size[1], self.grid_cols)):
                        managed.append(f"intersection_{r}_{c}")
                self.district_to_intersections[did] = managed

        # Define spaces -- single-agent wrapper uses intersection obs/action
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_phases)

        # Internal state (initialized in reset)
        self.rng: Optional[np.random.RandomState] = None
        self.current_step = 0
        self.queue_lengths: Optional[np.ndarray] = None  # (intersections, 4_approaches, lanes)
        self.phases: Optional[np.ndarray] = None  # current phase per intersection
        self.phase_durations: Optional[np.ndarray] = None
        self.waiting_times: Optional[np.ndarray] = None
        self.throughput_counter: Optional[np.ndarray] = None
        self.coordination_signals: Optional[np.ndarray] = None

        # Episode metrics
        self.episode_total_waiting = 0.0
        self.episode_total_throughput = 0.0
        self.episode_total_reward = 0.0

        logger.info(
            f"SyntheticTrafficEnvironment: {self.grid_rows}x{self.grid_cols} grid, "
            f"{self.num_intersections} intersections, {self.num_districts} districts"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Reset options.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self.rng = np.random.RandomState(effective_seed)

        self.current_step = 0

        # Initialize queues: (num_intersections, 4 approaches, num_lanes)
        self.queue_lengths = self.rng.uniform(
            0, 5, size=(self.num_intersections, 4, self.num_lanes)
        ).astype(np.float32)

        # Traffic light phases (one per intersection)
        self.phases = self.rng.randint(0, self.num_phases, size=self.num_intersections)
        self.phase_durations = np.zeros(self.num_intersections, dtype=np.float32)

        # Waiting times per intersection
        self.waiting_times = np.zeros(self.num_intersections, dtype=np.float32)

        # Throughput counter
        self.throughput_counter = np.zeros(self.num_intersections, dtype=np.float32)

        # Coordination signals from district agents
        self.coordination_signals = np.zeros(
            (self.num_intersections,), dtype=np.float32
        )

        # Episode metrics
        self.episode_total_waiting = 0.0
        self.episode_total_throughput = 0.0
        self.episode_total_reward = 0.0

        obs = self._get_observation(0)
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        For the single-agent wrapper, this applies the action to a round-robin
        selected intersection and advances the simulation.

        Args:
            action: Discrete action (traffic light phase).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Apply action to a round-robin intersection
        agent_idx = self.current_step % self.num_intersections
        self._apply_intersection_action(agent_idx, action)

        # If we've cycled through all intersections, advance simulation
        if agent_idx == self.num_intersections - 1:
            self._simulate_traffic_step()

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation(self.current_step % self.num_intersections)
        reward = self._compute_reward(agent_idx)
        info = self._get_info()

        self.episode_total_reward += reward

        return obs, reward, terminated, truncated, info

    def step_multi_agent(
        self, action_dict: Dict[str, Union[int, np.ndarray]]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Multi-agent step interface.

        Args:
            action_dict: Dictionary mapping agent IDs to actions.

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos).
        """
        # Apply intersection actions
        for i, iid in enumerate(self.intersection_ids):
            if iid in action_dict:
                action = action_dict[iid]
                if isinstance(action, np.ndarray):
                    action = int(action.item()) if action.size == 1 else int(action[0])
                self._apply_intersection_action(i, action)

        # Apply district actions (coordination signals)
        for did in self.district_ids:
            if did in action_dict:
                district_action = action_dict[did]
                if isinstance(district_action, (int, np.integer)):
                    district_action = np.zeros(self.high_level_action_dim)
                managed = self.district_to_intersections.get(did, [])
                for j, iid in enumerate(managed):
                    idx = self.intersection_ids.index(iid)
                    if j < len(district_action):
                        self.coordination_signals[idx] = float(district_action[j])

        # Advance simulation
        self._simulate_traffic_step()
        self.current_step += 1

        terminated = self.current_step >= self.max_steps

        # Build per-agent outputs
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for i, iid in enumerate(self.intersection_ids):
            observations[iid] = self._get_observation(i)
            rewards[iid] = self._compute_reward(i)
            terminateds[iid] = terminated
            truncateds[iid] = False
            infos[iid] = {"step": self.current_step}

        for did in self.district_ids:
            observations[did] = self._get_district_observation(did)
            rewards[did] = self._compute_district_reward(did)
            terminateds[did] = terminated
            truncateds[did] = False
            infos[did] = {"step": self.current_step}

        # Add __all__ keys for multi-agent termination
        terminateds["__all__"] = terminated
        truncateds["__all__"] = False

        total_reward = sum(r for r in rewards.values())
        self.episode_total_reward += total_reward

        return observations, rewards, terminateds, truncateds, infos

    def _apply_intersection_action(self, intersection_idx: int, action: int) -> None:
        """Apply a traffic light phase action to an intersection.

        Args:
            intersection_idx: Index of the intersection.
            action: Phase index to switch to.
        """
        action = int(action) % self.num_phases
        old_phase = self.phases[intersection_idx]

        if action != old_phase:
            # Phase switch: reset phase duration, add yellow penalty
            self.phases[intersection_idx] = action
            self.phase_durations[intersection_idx] = 0.0
        else:
            self.phase_durations[intersection_idx] += 1.0

    def _simulate_traffic_step(self) -> None:
        """Simulate one step of traffic dynamics.

        This is a simplified traffic model where:
        - Vehicles arrive at random rates influenced by time-of-day patterns
        - Green phases allow vehicles to depart (reduce queues)
        - Red phases cause vehicles to queue up
        - Coordination signals can modulate departure rates
        """
        # Time-of-day demand pattern (simple sinusoidal)
        time_factor = 1.0 + 0.5 * np.sin(
            2 * np.pi * self.current_step / self.max_steps
        )

        for i in range(self.num_intersections):
            phase = self.phases[i]

            for approach in range(4):
                for lane in range(self.num_lanes):
                    # Vehicle arrivals (Poisson-like)
                    arrival_rate = self.base_arrival_rate * time_factor
                    arrival_rate *= self.rng.uniform(0.5, 1.5)
                    arrivals = self.rng.poisson(arrival_rate)

                    # Vehicle departures depend on phase
                    # Phase 0: approaches 0,2 green; Phase 1: approaches 1,3 green
                    # Phase 2: approaches 0,2 left turn; Phase 3: approaches 1,3 left turn
                    is_green = False
                    if phase == 0 and approach in (0, 2):
                        is_green = True
                    elif phase == 1 and approach in (1, 3):
                        is_green = True
                    elif phase == 2 and approach in (0, 2) and lane == 0:
                        is_green = True
                    elif phase == 3 and approach in (1, 3) and lane == 0:
                        is_green = True

                    if is_green:
                        # Departure rate modified by coordination signal
                        coord_bonus = max(0.0, self.coordination_signals[i] * 0.2)
                        departure_rate = 2.0 + coord_bonus
                        departures = min(
                            self.queue_lengths[i, approach, lane],
                            self.rng.poisson(departure_rate),
                        )
                    else:
                        departures = 0.0

                    # Update queue
                    self.queue_lengths[i, approach, lane] = max(
                        0.0,
                        self.queue_lengths[i, approach, lane] + arrivals - departures,
                    )

            # Update waiting time (proportional to total queue)
            total_queue = self.queue_lengths[i].sum()
            self.waiting_times[i] = total_queue * 0.5

            # Update throughput
            throughput_this_step = max(0.0, self.rng.poisson(1.0) * (1.0 if total_queue < 20 else 0.5))
            self.throughput_counter[i] += throughput_this_step

        # Update episode metrics
        self.episode_total_waiting += float(self.waiting_times.sum())
        self.episode_total_throughput += float(self.throughput_counter.sum())

    def _get_observation(self, intersection_idx: int) -> np.ndarray:
        """Get observation for a single intersection agent.

        Args:
            intersection_idx: Index of the intersection.

        Returns:
            Observation vector of shape (obs_dim,).
        """
        obs_parts = []

        # Queue lengths per approach and lane (4 * num_lanes features)
        queues = self.queue_lengths[intersection_idx].flatten()
        obs_parts.append(queues / 20.0)  # normalize

        # Current phase (one-hot)
        phase_onehot = np.zeros(self.num_phases, dtype=np.float32)
        phase_onehot[self.phases[intersection_idx]] = 1.0
        obs_parts.append(phase_onehot)

        # Phase duration (normalized)
        obs_parts.append(
            np.array([self.phase_durations[intersection_idx] / 60.0], dtype=np.float32)
        )

        # Waiting time (normalized)
        obs_parts.append(
            np.array([self.waiting_times[intersection_idx] / 100.0], dtype=np.float32)
        )

        # Coordination signal received
        obs_parts.append(
            np.array([self.coordination_signals[intersection_idx]], dtype=np.float32)
        )

        # Neighbor info (up to 4 neighbors: up, down, left, right)
        row = intersection_idx // self.grid_cols
        col = intersection_idx % self.grid_cols
        neighbors = [
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)
        ]
        for nr, nc in neighbors:
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                nidx = nr * self.grid_cols + nc
                obs_parts.append(
                    np.array(
                        [
                            self.queue_lengths[nidx].sum() / 60.0,
                            self.waiting_times[nidx] / 100.0,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                obs_parts.append(np.zeros(2, dtype=np.float32))

        # Time feature
        obs_parts.append(
            np.array([self.current_step / self.max_steps], dtype=np.float32)
        )

        raw_obs = np.concatenate(obs_parts)

        # Pad or truncate to obs_dim
        if len(raw_obs) < self.obs_dim:
            raw_obs = np.concatenate(
                [raw_obs, np.zeros(self.obs_dim - len(raw_obs), dtype=np.float32)]
            )
        else:
            raw_obs = raw_obs[: self.obs_dim]

        return raw_obs.astype(np.float32)

    def _get_district_observation(self, district_id: str) -> np.ndarray:
        """Get observation for a district agent.

        Args:
            district_id: District agent ID.

        Returns:
            Observation vector of shape (high_level_obs_dim,).
        """
        managed = self.district_to_intersections.get(district_id, [])
        obs_parts = []

        total_queue = 0.0
        total_waiting = 0.0
        total_throughput = 0.0

        for iid in managed:
            idx = self.intersection_ids.index(iid)
            total_queue += self.queue_lengths[idx].sum()
            total_waiting += self.waiting_times[idx]
            total_throughput += self.throughput_counter[idx]

        obs_parts.append(np.array([
            total_queue / max(1.0, len(managed) * 60.0),
            total_waiting / max(1.0, len(managed) * 100.0),
            total_throughput / max(1.0, len(managed) * 1000.0),
            len(managed) / self.num_intersections,
            self.current_step / self.max_steps,
        ], dtype=np.float32))

        raw_obs = np.concatenate(obs_parts)

        # Pad or truncate
        if len(raw_obs) < self.high_level_obs_dim:
            raw_obs = np.concatenate(
                [raw_obs, np.zeros(self.high_level_obs_dim - len(raw_obs), dtype=np.float32)]
            )
        else:
            raw_obs = raw_obs[: self.high_level_obs_dim]

        return raw_obs.astype(np.float32)

    def _compute_reward(self, intersection_idx: int) -> float:
        """Compute reward for an intersection agent.

        Reward combines:
        - Negative waiting time penalty
        - Positive throughput bonus
        - Coordination bonus

        Args:
            intersection_idx: Index of the intersection.

        Returns:
            Scalar reward.
        """
        queue_total = float(self.queue_lengths[intersection_idx].sum())
        waiting = float(self.waiting_times[intersection_idx])

        # Waiting time penalty (normalized)
        waiting_penalty = -waiting / 100.0

        # Throughput reward (inversely proportional to queue)
        throughput_reward = max(0.0, 1.0 - queue_total / 60.0) * 0.5

        # Coordination bonus
        coord_bonus = 0.1 * float(self.coordination_signals[intersection_idx])

        reward = waiting_penalty + throughput_reward + coord_bonus
        return float(np.clip(reward, -5.0, 5.0))

    def _compute_district_reward(self, district_id: str) -> float:
        """Compute reward for a district agent.

        Args:
            district_id: District agent ID.

        Returns:
            Scalar reward (average of managed intersection rewards).
        """
        managed = self.district_to_intersections.get(district_id, [])
        if not managed:
            return 0.0

        total_reward = 0.0
        for iid in managed:
            idx = self.intersection_ids.index(iid)
            total_reward += self._compute_reward(idx)

        return total_reward / len(managed)

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info.

        Returns:
            Info dictionary with episode metrics.
        """
        return {
            "step": self.current_step,
            "total_queue": float(self.queue_lengths.sum()) if self.queue_lengths is not None else 0.0,
            "total_waiting": float(self.waiting_times.sum()) if self.waiting_times is not None else 0.0,
            "total_throughput": float(self.throughput_counter.sum()) if self.throughput_counter is not None else 0.0,
            "episode_total_waiting": self.episode_total_waiting,
            "episode_total_throughput": self.episode_total_throughput,
            "episode_total_reward": self.episode_total_reward,
        }

    def close(self) -> None:
        """Clean up resources."""
        pass

    def render(self) -> None:
        """Render the environment (no-op for training)."""
        pass
