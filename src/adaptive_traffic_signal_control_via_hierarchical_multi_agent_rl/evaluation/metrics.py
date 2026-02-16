"""Evaluation metrics for traffic control system performance."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import traci
from scipy import stats

from ..data.loader import SUMODataLoader
from ..training.trainer import TrafficEnvironment
from ..utils.config import Config

logger = logging.getLogger(__name__)


class TrafficMetrics:
    """Comprehensive traffic control evaluation metrics."""

    def __init__(self, config: Config) -> None:
        """Initialize traffic metrics evaluator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.metrics_history: List[Dict[str, float]] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None

    def evaluate_episode(
        self,
        environment: TrafficEnvironment,
        agent: Any,
        episode_length: int = 3600
    ) -> Dict[str, float]:
        """Evaluate a single episode.

        Args:
            environment: Traffic environment.
            agent: Trained agent or policy.
            episode_length: Episode length in seconds.

        Returns:
            Dictionary of episode metrics.
        """
        logger.info("Starting episode evaluation...")

        # Reset environment
        obs, _ = environment.reset()

        # Track metrics during episode
        episode_data = {
            "waiting_times": [],
            "throughput": [],
            "fuel_consumption": [],
            "coordination_events": 0,
            "phase_changes": {},
            "queue_lengths": []
        }

        terminated = {agent_id: False for agent_id in obs.keys()}
        step = 0

        while not all(terminated.values()) and step < episode_length:
            # Get actions from agent
            if hasattr(agent, 'predict'):
                # Stable-Baselines3 style
                actions = {}
                for agent_id, agent_obs in obs.items():
                    action, _ = agent.predict(agent_obs, deterministic=True)
                    actions[agent_id] = action
            else:
                # Custom agent style
                actions = agent.predict(obs, deterministic=True)

            # Step environment
            obs, rewards, terminated, truncated, infos = environment.step(actions)

            # Collect metrics
            self._collect_step_metrics(environment, episode_data, infos)

            step += 1

            if step % 100 == 0:
                logger.debug(f"Episode step {step}/{episode_length}")

        # Calculate final episode metrics
        episode_metrics = self._calculate_episode_metrics(episode_data, step)
        self.metrics_history.append(episode_metrics)

        logger.info(f"Episode evaluation completed. Mean waiting time: "
                   f"{episode_metrics['average_wait_time']:.2f}s")

        return episode_metrics

    def _collect_step_metrics(
        self,
        environment: TrafficEnvironment,
        episode_data: Dict[str, List[Any]],
        infos: Dict[str, Any]
    ) -> None:
        """Collect metrics for a single step.

        Args:
            environment: Traffic environment.
            episode_data: Episode data collection dictionary.
            infos: Step info from environment.
        """
        try:
            # Switch to SUMO connection
            if environment.sumo_label:
                traci.switch(environment.sumo_label)

                # Collect waiting times
                waiting_times = []
                for vehicle_id in traci.vehicle.getIDList():
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    waiting_times.append(waiting_time)

                episode_data["waiting_times"].extend(waiting_times)

                # Collect throughput (completed trips this step)
                arrived_vehicles = traci.simulation.getArrivedNumber()
                episode_data["throughput"].append(arrived_vehicles)

                # Collect fuel consumption (simplified)
                active_vehicles = len(traci.vehicle.getIDList())
                fuel_consumption = active_vehicles * 0.001  # Simplified fuel model
                episode_data["fuel_consumption"].append(fuel_consumption)

                # Collect queue lengths
                total_queue_length = 0
                for agent_id in environment.intersection_agents:
                    intersection_data = environment.intersection_agents[agent_id]["intersection_data"]
                    queue_data = environment._get_queue_data(intersection_data)
                    agent_queue_length = sum(queue_data.get("lane_queues", {}).values())
                    total_queue_length += agent_queue_length

                episode_data["queue_lengths"].append(total_queue_length)

                # Track coordination events
                if any("coordination" in str(info) for info in infos.values()):
                    episode_data["coordination_events"] += 1

        except Exception as e:
            logger.warning(f"Failed to collect step metrics: {e}")

    def _calculate_episode_metrics(
        self,
        episode_data: Dict[str, List[Any]],
        total_steps: int
    ) -> Dict[str, float]:
        """Calculate metrics for completed episode.

        Args:
            episode_data: Collected episode data.
            total_steps: Total number of steps.

        Returns:
            Dictionary of calculated metrics.
        """
        metrics = {}

        # Average waiting time
        all_waiting_times = episode_data["waiting_times"]
        metrics["average_wait_time"] = np.mean(all_waiting_times) if all_waiting_times else 0.0
        metrics["max_wait_time"] = np.max(all_waiting_times) if all_waiting_times else 0.0
        metrics["wait_time_std"] = np.std(all_waiting_times) if all_waiting_times else 0.0

        # Throughput metrics
        total_throughput = sum(episode_data["throughput"])
        metrics["total_throughput"] = total_throughput
        metrics["throughput_rate"] = total_throughput / max(total_steps, 1)

        # Fuel consumption
        total_fuel = sum(episode_data["fuel_consumption"])
        metrics["total_fuel_consumption"] = total_fuel
        metrics["fuel_efficiency"] = total_fuel / max(total_throughput, 1)

        # Queue metrics
        queue_lengths = episode_data["queue_lengths"]
        metrics["average_queue_length"] = np.mean(queue_lengths) if queue_lengths else 0.0
        metrics["max_queue_length"] = np.max(queue_lengths) if queue_lengths else 0.0

        # Coordination efficiency
        coordination_events = episode_data["coordination_events"]
        metrics["coordination_efficiency"] = coordination_events / max(total_steps, 1)

        # Derived metrics
        metrics["travel_time_index"] = self._calculate_travel_time_index(metrics)
        metrics["level_of_service"] = self._calculate_level_of_service(metrics)

        return metrics

    def _calculate_travel_time_index(self, metrics: Dict[str, float]) -> float:
        """Calculate travel time index.

        Args:
            metrics: Episode metrics.

        Returns:
            Travel time index (ratio of actual to free-flow travel time).
        """
        # Simplified calculation based on waiting time and throughput
        base_travel_time = 120.0  # seconds (free-flow time across network)
        additional_time = metrics["average_wait_time"]

        travel_time_index = (base_travel_time + additional_time) / base_travel_time
        return min(travel_time_index, 5.0)  # Cap at 5.0

    def _calculate_level_of_service(self, metrics: Dict[str, float]) -> str:
        """Calculate Level of Service (LOS).

        Args:
            metrics: Episode metrics.

        Returns:
            Level of Service grade (A-F).
        """
        # Based on average waiting time
        wait_time = metrics["average_wait_time"]

        if wait_time < 10:
            return "A"
        elif wait_time < 20:
            return "B"
        elif wait_time < 35:
            return "C"
        elif wait_time < 55:
            return "D"
        elif wait_time < 80:
            return "E"
        else:
            return "F"

    def evaluate_baseline(
        self,
        environment: TrafficEnvironment,
        episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate baseline (fixed-time) traffic control.

        Args:
            environment: Traffic environment.
            episodes: Number of evaluation episodes.

        Returns:
            Baseline performance metrics.
        """
        logger.info(f"Evaluating baseline performance over {episodes} episodes...")

        baseline_results = []

        for episode in range(episodes):
            obs, _ = environment.reset()

            episode_data = {
                "waiting_times": [],
                "throughput": [],
                "fuel_consumption": [],
                "coordination_events": 0,
                "queue_lengths": []
            }

            # Fixed-time control (no agent actions, use default SUMO timing)
            terminated = {agent_id: False for agent_id in obs.keys()}
            step = 0
            max_steps = 3600

            while not all(terminated.values()) and step < max_steps:
                # No actions (use default signal timing)
                actions = {agent_id: 0 for agent_id in obs.keys()}  # Default action

                obs, rewards, terminated, truncated, infos = environment.step(actions)
                self._collect_step_metrics(environment, episode_data, infos)
                step += 1

            episode_metrics = self._calculate_episode_metrics(episode_data, step)
            baseline_results.append(episode_metrics)

            logger.debug(f"Baseline episode {episode + 1}/{episodes} completed")

        # Aggregate baseline results
        self.baseline_metrics = self._aggregate_metrics(baseline_results)

        logger.info(f"Baseline evaluation completed. Mean waiting time: "
                   f"{self.baseline_metrics['average_wait_time']:.2f}s")

        return self.baseline_metrics

    def _aggregate_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple episodes.

        Args:
            results: List of episode results.

        Returns:
            Aggregated metrics with statistics.
        """
        if not results:
            return {}

        aggregated = {}
        metrics_keys = results[0].keys()

        for key in metrics_keys:
            values = [result[key] for result in results]

            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

        return aggregated

    def calculate_improvement_metrics(self) -> Dict[str, float]:
        """Calculate improvement metrics compared to baseline.

        Returns:
            Dictionary of improvement metrics.
        """
        if not self.metrics_history or not self.baseline_metrics:
            logger.warning("Cannot calculate improvements: missing baseline or episode metrics")
            return {}

        # Get latest episode metrics
        latest_metrics = self.metrics_history[-1]
        improvements = {}

        # Wait time reduction
        wait_time_improvement = (
            (self.baseline_metrics["average_wait_time"] - latest_metrics["average_wait_time"]) /
            self.baseline_metrics["average_wait_time"]
        )
        improvements["average_wait_time_reduction"] = wait_time_improvement

        # Throughput improvement
        throughput_improvement = (
            (latest_metrics["throughput_rate"] - self.baseline_metrics["throughput_rate"]) /
            max(self.baseline_metrics["throughput_rate"], 1e-6)
        )
        improvements["throughput_improvement"] = throughput_improvement

        # Fuel efficiency improvement
        fuel_improvement = (
            (self.baseline_metrics["fuel_efficiency"] - latest_metrics["fuel_efficiency"]) /
            self.baseline_metrics["fuel_efficiency"]
        )
        improvements["fuel_efficiency_improvement"] = fuel_improvement

        # Overall performance score
        improvements["overall_performance_score"] = (
            0.5 * wait_time_improvement +
            0.3 * throughput_improvement +
            0.2 * fuel_improvement
        )

        return improvements

    def analyze_coordination_effectiveness(self) -> Dict[str, float]:
        """Analyze coordination effectiveness between agents.

        Returns:
            Coordination analysis metrics.
        """
        if len(self.metrics_history) < 2:
            return {"coordination_efficiency": 0.0}

        # Analyze coordination events over time
        coordination_events = [m.get("coordination_efficiency", 0) for m in self.metrics_history]

        # Calculate trends
        if len(coordination_events) >= 10:
            recent_coordination = np.mean(coordination_events[-10:])
            early_coordination = np.mean(coordination_events[:10])
            coordination_trend = recent_coordination - early_coordination
        else:
            coordination_trend = 0.0

        # Analyze correlation between coordination and performance
        wait_times = [m.get("average_wait_time", 0) for m in self.metrics_history]

        if len(coordination_events) >= 5 and len(wait_times) >= 5:
            correlation, p_value = stats.pearsonr(coordination_events, wait_times)
            coordination_effectiveness = -correlation if p_value < 0.05 else 0.0
        else:
            coordination_effectiveness = 0.0

        return {
            "coordination_efficiency": np.mean(coordination_events),
            "coordination_trend": coordination_trend,
            "coordination_effectiveness": coordination_effectiveness,
        }

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Returns:
            Detailed performance report.
        """
        logger.info("Generating comprehensive performance report...")

        if not self.metrics_history:
            return {"error": "No episode metrics available"}

        report = {
            "summary": {},
            "improvements": {},
            "coordination": {},
            "trends": {},
            "target_achievements": {}
        }

        # Summary statistics
        latest_metrics = self.metrics_history[-1]
        report["summary"] = latest_metrics.copy()

        # Improvement analysis
        if self.baseline_metrics:
            report["improvements"] = self.calculate_improvement_metrics()

        # Coordination analysis
        report["coordination"] = self.analyze_coordination_effectiveness()

        # Trend analysis
        report["trends"] = self._analyze_trends()

        # Target achievement analysis
        report["target_achievements"] = self._check_target_achievements(report)

        logger.info("Performance report generated successfully")
        return report

    def _analyze_trends(self) -> Dict[str, float]:
        """Analyze performance trends over episodes.

        Returns:
            Trend analysis metrics.
        """
        if len(self.metrics_history) < 5:
            return {}

        trends = {}
        metrics_to_analyze = ["average_wait_time", "throughput_rate", "coordination_efficiency"]

        for metric in metrics_to_analyze:
            values = [m.get(metric, 0) for m in self.metrics_history]

            if len(values) >= 5:
                # Linear trend analysis
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

                trends[f"{metric}_slope"] = slope
                trends[f"{metric}_r_squared"] = r_value ** 2
                trends[f"{metric}_improving"] = (slope < 0 if "wait_time" in metric else slope > 0)

        return trends

    def _check_target_achievements(self, report: Dict[str, Any]) -> Dict[str, bool]:
        """Check if target metrics are achieved.

        Args:
            report: Performance report.

        Returns:
            Target achievement status.
        """
        improvements = report.get("improvements", {})
        coordination = report.get("coordination", {})

        target_metrics = self.config.target_metrics

        achievements = {}

        # Check wait time reduction target
        wait_time_reduction = improvements.get("average_wait_time_reduction", 0)
        achievements["wait_time_target"] = wait_time_reduction >= target_metrics.average_wait_time_reduction

        # Check throughput improvement target
        throughput_improvement = improvements.get("throughput_improvement", 0)
        achievements["throughput_target"] = throughput_improvement >= target_metrics.throughput_improvement

        # Check coordination efficiency target
        coordination_efficiency = coordination.get("coordination_efficiency", 0)
        achievements["coordination_target"] = coordination_efficiency >= target_metrics.coordination_efficiency

        # Overall achievement
        achievements["all_targets_met"] = all(achievements.values())

        return achievements

    def visualize_performance(self, save_path: Optional[str] = None) -> None:
        """Create performance visualization plots.

        Args:
            save_path: Optional path to save plots.
        """
        if not self.metrics_history:
            logger.warning("No metrics available for visualization")
            return

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Traffic Control Performance Analysis", fontsize=16)

        # Plot 1: Waiting time over episodes
        episodes = range(1, len(self.metrics_history) + 1)
        wait_times = [m["average_wait_time"] for m in self.metrics_history]

        axes[0, 0].plot(episodes, wait_times, 'b-', linewidth=2, label="Agent")
        if self.baseline_metrics:
            baseline_wait = self.baseline_metrics["average_wait_time"]
            axes[0, 0].axhline(y=baseline_wait, color='r', linestyle='--',
                              linewidth=2, label="Baseline")

        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Average Waiting Time (s)")
        axes[0, 0].set_title("Waiting Time Performance")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Throughput over episodes
        throughput_rates = [m["throughput_rate"] for m in self.metrics_history]

        axes[0, 1].plot(episodes, throughput_rates, 'g-', linewidth=2, label="Agent")
        if self.baseline_metrics:
            baseline_throughput = self.baseline_metrics["throughput_rate"]
            axes[0, 1].axhline(y=baseline_throughput, color='r', linestyle='--',
                              linewidth=2, label="Baseline")

        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Throughput Rate (vehicles/step)")
        axes[0, 1].set_title("Throughput Performance")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Coordination efficiency
        coordination_eff = [m.get("coordination_efficiency", 0) for m in self.metrics_history]

        axes[1, 0].plot(episodes, coordination_eff, 'm-', linewidth=2)
        target_coord = self.config.target_metrics.coordination_efficiency
        axes[1, 0].axhline(y=target_coord, color='orange', linestyle='--',
                          linewidth=2, label=f"Target ({target_coord})")

        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Coordination Efficiency")
        axes[1, 0].set_title("Agent Coordination")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Performance distribution
        if len(wait_times) >= 10:
            axes[1, 1].hist(wait_times, bins=15, alpha=0.7, color='skyblue',
                           edgecolor='black', label="Agent")

            if self.baseline_metrics:
                baseline_wait = self.baseline_metrics["average_wait_time"]
                axes[1, 1].axvline(x=baseline_wait, color='red', linestyle='--',
                                  linewidth=2, label="Baseline")

            axes[1, 1].set_xlabel("Average Waiting Time (s)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Performance Distribution")
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, "Insufficient data\nfor distribution plot",
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Performance Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")

        plt.show()


class TransferLearningEvaluator:
    """Evaluator for transfer learning capabilities."""

    def __init__(self, config: Config) -> None:
        """Initialize transfer learning evaluator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.source_performance: Dict[str, Dict[str, float]] = {}
        self.target_performance: Dict[str, Dict[str, float]] = {}

    def evaluate_source_performance(
        self,
        agent: Any,
        source_scenario: str,
        episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate agent performance on source scenario.

        Args:
            agent: Trained agent.
            source_scenario: Source scenario name.
            episodes: Number of evaluation episodes.

        Returns:
            Source performance metrics.
        """
        logger.info(f"Evaluating source performance on {source_scenario}...")

        # Create environment for source scenario
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": source_scenario
        }

        env = TrafficEnvironment(env_config)
        metrics_evaluator = TrafficMetrics(self.config)

        # Evaluate multiple episodes
        episode_results = []
        for episode in range(episodes):
            episode_metrics = metrics_evaluator.evaluate_episode(env, agent)
            episode_results.append(episode_metrics)

        # Aggregate results
        source_metrics = metrics_evaluator._aggregate_metrics(episode_results)
        self.source_performance[source_scenario] = source_metrics

        env.close()

        logger.info(f"Source evaluation completed for {source_scenario}")
        return source_metrics

    def evaluate_target_performance(
        self,
        agent: Any,
        target_scenario: str,
        source_scenario: str,
        episodes: int = 10,
        fine_tune_steps: int = 0
    ) -> Dict[str, float]:
        """Evaluate agent transfer performance on target scenario.

        Args:
            agent: Pre-trained agent from source scenario.
            target_scenario: Target scenario name.
            source_scenario: Source scenario name (for reference).
            episodes: Number of evaluation episodes.
            fine_tune_steps: Optional fine-tuning steps on target.

        Returns:
            Target performance metrics.
        """
        logger.info(f"Evaluating transfer performance: {source_scenario} → {target_scenario}")

        # Create environment for target scenario
        env_config = {
            "rl_config": self.config.to_dict(),
            "scenario": target_scenario
        }

        env = TrafficEnvironment(env_config)

        # Optional fine-tuning on target scenario
        if fine_tune_steps > 0:
            logger.info(f"Fine-tuning agent for {fine_tune_steps} steps on target scenario")
            agent = self._fine_tune_agent(agent, env, fine_tune_steps)

        # Evaluate on target scenario
        metrics_evaluator = TrafficMetrics(self.config)
        episode_results = []

        for episode in range(episodes):
            episode_metrics = metrics_evaluator.evaluate_episode(env, agent)
            episode_results.append(episode_metrics)

        # Aggregate results
        target_metrics = metrics_evaluator._aggregate_metrics(episode_results)

        transfer_key = f"{source_scenario}_to_{target_scenario}"
        self.target_performance[transfer_key] = target_metrics

        env.close()

        logger.info(f"Transfer evaluation completed: {transfer_key}")
        return target_metrics

    def _fine_tune_agent(self, agent: Any, env: TrafficEnvironment, steps: int) -> Any:
        """Fine-tune agent on target scenario.

        Args:
            agent: Pre-trained agent.
            env: Target environment.
            steps: Number of fine-tuning steps.

        Returns:
            Fine-tuned agent.
        """
        # This is a simplified implementation
        # In practice, you would implement proper fine-tuning
        logger.info(f"Fine-tuning agent for {steps} steps (simplified implementation)")

        # For demonstration, we just return the original agent
        # In a real implementation, you would:
        # 1. Continue training the agent on the target environment
        # 2. Use a lower learning rate for fine-tuning
        # 3. Implement domain adaptation techniques if needed

        return agent

    def calculate_transfer_metrics(
        self,
        source_scenario: str,
        target_scenario: str
    ) -> Dict[str, float]:
        """Calculate transfer learning metrics.

        Args:
            source_scenario: Source scenario name.
            target_scenario: Target scenario name.

        Returns:
            Transfer learning metrics.
        """
        transfer_key = f"{source_scenario}_to_{target_scenario}"

        if source_scenario not in self.source_performance:
            raise ValueError(f"Source performance not available for {source_scenario}")

        if transfer_key not in self.target_performance:
            raise ValueError(f"Target performance not available for {transfer_key}")

        source_metrics = self.source_performance[source_scenario]
        target_metrics = self.target_performance[transfer_key]

        transfer_metrics = {}

        # Transfer learning retention (how much performance is retained)
        source_wait_time = source_metrics["average_wait_time"]
        target_wait_time = target_metrics["average_wait_time"]

        # Higher retention is better (closer to 1.0 means better transfer)
        if source_wait_time > 0:
            wait_time_retention = min(source_wait_time / target_wait_time, 2.0)
        else:
            wait_time_retention = 1.0

        transfer_metrics["transfer_learning_retention"] = wait_time_retention

        # Performance degradation
        degradation = (target_wait_time - source_wait_time) / max(source_wait_time, 1.0)
        transfer_metrics["performance_degradation"] = degradation

        # Throughput transfer
        source_throughput = source_metrics.get("throughput_rate", 0)
        target_throughput = target_metrics.get("throughput_rate", 0)

        if source_throughput > 0:
            throughput_retention = target_throughput / source_throughput
        else:
            throughput_retention = 1.0

        transfer_metrics["throughput_retention"] = throughput_retention

        # Overall transfer success score
        transfer_score = (
            0.6 * wait_time_retention +
            0.4 * throughput_retention
        )
        transfer_metrics["overall_transfer_score"] = min(transfer_score, 1.0)

        return transfer_metrics

    def evaluate_cross_scenario_transfer(
        self,
        agent: Any,
        scenario_pairs: List[Tuple[str, str]],
        episodes_per_scenario: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate transfer learning across multiple scenario pairs.

        Args:
            agent: Trained agent.
            scenario_pairs: List of (source, target) scenario pairs.
            episodes_per_scenario: Episodes to evaluate per scenario.

        Returns:
            Comprehensive transfer learning results.
        """
        logger.info(f"Evaluating cross-scenario transfer for {len(scenario_pairs)} pairs")

        transfer_results = {}

        for source_scenario, target_scenario in scenario_pairs:
            try:
                # Evaluate source performance if not already done
                if source_scenario not in self.source_performance:
                    self.evaluate_source_performance(agent, source_scenario, episodes_per_scenario)

                # Evaluate transfer to target
                self.evaluate_target_performance(
                    agent, target_scenario, source_scenario, episodes_per_scenario
                )

                # Calculate transfer metrics
                transfer_metrics = self.calculate_transfer_metrics(source_scenario, target_scenario)

                transfer_key = f"{source_scenario}_to_{target_scenario}"
                transfer_results[transfer_key] = transfer_metrics

                logger.info(f"Transfer evaluation completed: {transfer_key}")

            except Exception as e:
                logger.error(f"Failed to evaluate transfer {source_scenario} → {target_scenario}: {e}")
                continue

        return transfer_results

    def generate_transfer_report(self) -> Dict[str, Any]:
        """Generate comprehensive transfer learning report.

        Returns:
            Transfer learning analysis report.
        """
        logger.info("Generating transfer learning report...")

        if not self.source_performance or not self.target_performance:
            return {"error": "Insufficient transfer evaluation data"}

        report = {
            "summary": {},
            "detailed_results": {},
            "target_achievements": {}
        }

        # Summary statistics
        all_transfer_scores = []
        all_retentions = []

        for transfer_key, target_perf in self.target_performance.items():
            source_scenario = transfer_key.split("_to_")[0]
            target_scenario = transfer_key.split("_to_")[1]

            if source_scenario in self.source_performance:
                transfer_metrics = self.calculate_transfer_metrics(source_scenario, target_scenario)

                all_transfer_scores.append(transfer_metrics["overall_transfer_score"])
                all_retentions.append(transfer_metrics["transfer_learning_retention"])

                report["detailed_results"][transfer_key] = transfer_metrics

        # Overall summary
        if all_transfer_scores:
            report["summary"] = {
                "mean_transfer_score": np.mean(all_transfer_scores),
                "mean_retention": np.mean(all_retentions),
                "min_transfer_score": np.min(all_transfer_scores),
                "max_transfer_score": np.max(all_transfer_scores),
                "std_transfer_score": np.std(all_transfer_scores),
                "num_evaluations": len(all_transfer_scores)
            }

            # Check target achievement
            target_retention = self.config.target_metrics.transfer_learning_retention
            retention_achieved = np.mean(all_retentions) >= target_retention

            report["target_achievements"] = {
                "transfer_learning_target": retention_achieved,
                "target_retention": target_retention,
                "achieved_retention": np.mean(all_retentions)
            }

        logger.info("Transfer learning report generated successfully")
        return report

    def visualize_transfer_results(self, save_path: Optional[str] = None) -> None:
        """Create transfer learning visualization plots.

        Args:
            save_path: Optional path to save plots.
        """
        if not self.target_performance:
            logger.warning("No transfer results available for visualization")
            return

        # Calculate all transfer metrics
        transfer_data = []
        for transfer_key in self.target_performance.keys():
            source_scenario = transfer_key.split("_to_")[0]
            target_scenario = transfer_key.split("_to_")[1]

            if source_scenario in self.source_performance:
                metrics = self.calculate_transfer_metrics(source_scenario, target_scenario)
                transfer_data.append({
                    "source": source_scenario,
                    "target": target_scenario,
                    "transfer_key": transfer_key,
                    **metrics
                })

        if not transfer_data:
            logger.warning("No valid transfer data for visualization")
            return

        df = pd.DataFrame(transfer_data)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Transfer Learning Analysis", fontsize=16)

        # Plot 1: Transfer retention scores
        transfer_keys = [d["transfer_key"] for d in transfer_data]
        retentions = [d["transfer_learning_retention"] for d in transfer_data]

        axes[0, 0].bar(range(len(transfer_keys)), retentions, alpha=0.7, color='skyblue')
        target_retention = self.config.target_metrics.transfer_learning_retention
        axes[0, 0].axhline(y=target_retention, color='red', linestyle='--',
                          linewidth=2, label=f"Target ({target_retention})")

        axes[0, 0].set_xlabel("Transfer Scenario")
        axes[0, 0].set_ylabel("Transfer Learning Retention")
        axes[0, 0].set_title("Transfer Learning Retention")
        axes[0, 0].set_xticks(range(len(transfer_keys)))
        axes[0, 0].set_xticklabels([k.replace("_to_", "→") for k in transfer_keys],
                                   rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Overall transfer scores
        transfer_scores = [d["overall_transfer_score"] for d in transfer_data]

        axes[0, 1].bar(range(len(transfer_keys)), transfer_scores, alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel("Transfer Scenario")
        axes[0, 1].set_ylabel("Overall Transfer Score")
        axes[0, 1].set_title("Overall Transfer Performance")
        axes[0, 1].set_xticks(range(len(transfer_keys)))
        axes[0, 1].set_xticklabels([k.replace("_to_", "→") for k in transfer_keys],
                                   rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Performance degradation
        degradations = [d["performance_degradation"] for d in transfer_data]

        axes[1, 0].bar(range(len(transfer_keys)), degradations, alpha=0.7,
                      color=['red' if d > 0 else 'green' for d in degradations])
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].set_xlabel("Transfer Scenario")
        axes[1, 0].set_ylabel("Performance Degradation")
        axes[1, 0].set_title("Performance Change in Transfer")
        axes[1, 0].set_xticks(range(len(transfer_keys)))
        axes[1, 0].set_xticklabels([k.replace("_to_", "→") for k in transfer_keys],
                                   rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Correlation between retention and transfer score
        if len(transfer_data) >= 3:
            axes[1, 1].scatter(retentions, transfer_scores, alpha=0.7, s=100, color='purple')

            # Add trend line
            z = np.polyfit(retentions, transfer_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(retentions), max(retentions), 100)
            axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8)

            axes[1, 1].set_xlabel("Transfer Learning Retention")
            axes[1, 1].set_ylabel("Overall Transfer Score")
            axes[1, 1].set_title("Retention vs. Transfer Score")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "Insufficient data\nfor correlation plot",
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Retention vs. Transfer Score")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Transfer learning plots saved to {save_path}")

        plt.show()