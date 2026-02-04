#!/usr/bin/env python3
"""Evaluation script for hierarchical multi-agent RL traffic control system.

This script provides comprehensive evaluation capabilities including:
- Performance evaluation against baseline
- Transfer learning assessment
- Visualization generation
- Statistical analysis
- Model comparison

Usage:
    python scripts/evaluate.py [OPTIONS]

Examples:
    # Basic evaluation
    python scripts/evaluate.py --model-path models/trained_model.pth

    # Evaluation with transfer learning
    python scripts/evaluate.py --model-path models/trained_model.pth --transfer-learning

    # Comparison between models
    python scripts/evaluate.py --model-path models/model1.pth --compare-with models/model2.pth

    # Generate visualizations
    python scripts/evaluate.py --model-path models/trained_model.pth --visualize --output-dir results/
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.utils.config import (
    load_config,
    setup_logging,
    set_random_seeds
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.evaluation.metrics import (
    TrafficMetrics,
    TransferLearningEvaluator
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer import (
    TrafficEnvironment
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import (
    HierarchicalTrafficAgent,
    TrafficEnvironmentWrapper
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical multi-agent RL traffic control system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model and configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Traffic scenario to evaluate on"
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy during evaluation"
    )

    # Baseline comparison
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline (fixed-time) evaluation"
    )

    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=5,
        help="Number of baseline evaluation episodes"
    )

    # Model comparison
    parser.add_argument(
        "--compare-with",
        type=str,
        nargs="+",
        default=None,
        help="Paths to additional models for comparison"
    )

    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=None,
        help="Names for models in comparison (same order as --compare-with)"
    )

    # Transfer learning
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Evaluate transfer learning capabilities"
    )

    parser.add_argument(
        "--source-scenarios",
        type=str,
        nargs="+",
        default=["manhattan_grid"],
        help="Source scenarios for transfer learning"
    )

    parser.add_argument(
        "--target-scenarios",
        type=str,
        nargs="+",
        default=["cologne"],
        help="Target scenarios for transfer learning"
    )

    parser.add_argument(
        "--fine-tune-steps",
        type=int,
        default=0,
        help="Fine-tuning steps for transfer learning"
    )

    # Output and visualization
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files"
    )

    parser.add_argument(
        "--plot-format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Format for saved plots"
    )

    # Analysis options
    parser.add_argument(
        "--statistical-test",
        action="store_true",
        help="Perform statistical significance tests"
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for statistical tests"
    )

    parser.add_argument(
        "--detailed-metrics",
        action="store_true",
        help="Collect detailed per-step metrics"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )

    return parser.parse_args()


def load_model(model_path: str, config: Any) -> Any:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        config: Configuration object.

    Returns:
        Loaded model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        RuntimeError: If model loading fails.
    """
    logger.info(f"Loading model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Create hierarchical agent
        hierarchical_agent = HierarchicalTrafficAgent(config)

        # Load checkpoint
        hierarchical_agent.load_checkpoint(model_path)

        # Create environment wrapper
        model = TrafficEnvironmentWrapper(config, hierarchical_agent)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def evaluate_model(
    model: Any,
    config: Any,
    scenario: str,
    episodes: int,
    deterministic: bool = True
) -> Dict[str, Any]:
    """Evaluate model performance.

    Args:
        model: Trained model.
        config: Configuration object.
        scenario: Scenario name.
        episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic policy.

    Returns:
        Evaluation results dictionary.
    """
    logger.info(f"Evaluating model on {scenario} for {episodes} episodes")

    # Create environment
    env_config = {
        "rl_config": config.to_dict(),
        "scenario": scenario
    }

    env = TrafficEnvironment(env_config)
    metrics_evaluator = TrafficMetrics(config)

    # Run evaluation episodes
    episode_results = []
    start_time = time.time()

    for episode in range(episodes):
        logger.debug(f"Running episode {episode + 1}/{episodes}")

        episode_metrics = metrics_evaluator.evaluate_episode(
            env, model, episode_length=config.environment.simulation_time
        )

        episode_results.append(episode_metrics)

        if (episode + 1) % max(1, episodes // 10) == 0:
            logger.info(f"Completed {episode + 1}/{episodes} episodes")

    evaluation_time = time.time() - start_time

    # Aggregate results
    aggregated_results = metrics_evaluator._aggregate_metrics(episode_results)

    # Add evaluation metadata
    aggregated_results.update({
        "scenario": scenario,
        "episodes": episodes,
        "deterministic": deterministic,
        "evaluation_time": evaluation_time,
        "episode_results": episode_results
    })

    env.close()

    logger.info(f"Model evaluation completed in {evaluation_time:.2f} seconds")
    return aggregated_results


def evaluate_baseline(
    config: Any,
    scenario: str,
    episodes: int
) -> Dict[str, Any]:
    """Evaluate baseline (fixed-time) performance.

    Args:
        config: Configuration object.
        scenario: Scenario name.
        episodes: Number of evaluation episodes.

    Returns:
        Baseline evaluation results.
    """
    logger.info(f"Evaluating baseline on {scenario} for {episodes} episodes")

    # Create environment
    env_config = {
        "rl_config": config.to_dict(),
        "scenario": scenario
    }

    env = TrafficEnvironment(env_config)
    metrics_evaluator = TrafficMetrics(config)

    # Evaluate baseline
    baseline_results = metrics_evaluator.evaluate_baseline(env, episodes)

    env.close()

    logger.info("Baseline evaluation completed")
    return baseline_results


def compare_models(
    model_results: List[Dict[str, Any]],
    model_names: List[str],
    config: Any
) -> Dict[str, Any]:
    """Compare multiple model results.

    Args:
        model_results: List of model evaluation results.
        model_names: List of model names.
        config: Configuration object.

    Returns:
        Comparison results dictionary.
    """
    logger.info(f"Comparing {len(model_results)} models")

    comparison = {
        "models": model_names,
        "metrics_comparison": {},
        "statistical_tests": {},
        "rankings": {}
    }

    # Key metrics to compare
    metrics_to_compare = [
        "average_wait_time",
        "throughput_rate",
        "coordination_efficiency",
        "total_fuel_consumption"
    ]

    # Compare metrics
    for metric in metrics_to_compare:
        values = []
        for result in model_results:
            if metric in result:
                values.append(result[metric])
            else:
                values.append(np.nan)

        comparison["metrics_comparison"][metric] = {
            "values": values,
            "best_model": model_names[np.nanargmin(values) if "time" in metric or "consumption" in metric
                                   else np.nanargmax(values)],
            "worst_model": model_names[np.nanargmax(values) if "time" in metric or "consumption" in metric
                                    else np.nanargmin(values)]
        }

    # Statistical tests (if episode data available)
    if all("episode_results" in result for result in model_results):
        from scipy import stats

        for metric in metrics_to_compare:
            metric_data = []
            for result in model_results:
                episode_values = [ep.get(metric, np.nan) for ep in result["episode_results"]]
                metric_data.append(episode_values)

            # Perform ANOVA if more than 2 models
            if len(metric_data) > 2:
                try:
                    f_stat, p_value = stats.f_oneway(*metric_data)
                    comparison["statistical_tests"][metric] = {
                        "test": "ANOVA",
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Statistical test failed for {metric}: {e}")

            # Pairwise t-tests
            elif len(metric_data) == 2:
                try:
                    t_stat, p_value = stats.ttest_ind(metric_data[0], metric_data[1])
                    comparison["statistical_tests"][metric] = {
                        "test": "t-test",
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Statistical test failed for {metric}: {e}")

    # Overall ranking
    wait_time_ranks = stats.rankdata([r.get("average_wait_time", float('inf')) for r in model_results])
    throughput_ranks = stats.rankdata([-r.get("throughput_rate", 0) for r in model_results])

    overall_ranks = (wait_time_ranks + throughput_ranks) / 2
    ranking_order = np.argsort(overall_ranks)

    comparison["rankings"] = {
        "overall": [model_names[i] for i in ranking_order],
        "wait_time": [model_names[i] for i in np.argsort(wait_time_ranks)],
        "throughput": [model_names[i] for i in np.argsort(throughput_ranks)]
    }

    logger.info("Model comparison completed")
    return comparison


def evaluate_transfer_learning(
    model: Any,
    config: Any,
    source_scenarios: List[str],
    target_scenarios: List[str],
    episodes_per_scenario: int = 5
) -> Dict[str, Any]:
    """Evaluate transfer learning capabilities.

    Args:
        model: Trained model.
        config: Configuration object.
        source_scenarios: List of source scenarios.
        target_scenarios: List of target scenarios.
        episodes_per_scenario: Episodes per scenario.

    Returns:
        Transfer learning evaluation results.
    """
    logger.info("Evaluating transfer learning capabilities")

    transfer_evaluator = TransferLearningEvaluator(config)

    # Create scenario pairs
    scenario_pairs = []
    for source in source_scenarios:
        for target in target_scenarios:
            if source != target:
                scenario_pairs.append((source, target))

    if not scenario_pairs:
        logger.warning("No valid transfer scenario pairs found")
        return {}

    # Evaluate transfer learning
    transfer_results = transfer_evaluator.evaluate_cross_scenario_transfer(
        model, scenario_pairs, episodes_per_scenario
    )

    # Generate transfer report
    transfer_report = transfer_evaluator.generate_transfer_report()

    logger.info("Transfer learning evaluation completed")
    return {
        "transfer_results": transfer_results,
        "transfer_report": transfer_report,
        "scenario_pairs": scenario_pairs
    }


def create_visualizations(
    results: Dict[str, Any],
    output_dir: str,
    save_plots: bool = False,
    plot_format: str = "png"
) -> None:
    """Create visualization plots.

    Args:
        results: Evaluation results.
        output_dir: Output directory.
        save_plots: Whether to save plots to files.
        plot_format: Format for saved plots.
    """
    logger.info("Creating visualizations")

    plt.style.use('seaborn-v0_8')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Performance comparison plot
    if "model_evaluation" in results and "baseline_evaluation" in results:
        model_result = results["model_evaluation"]
        baseline_result = results["baseline_evaluation"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model vs Baseline Performance Comparison", fontsize=16)

        # Waiting time comparison
        metrics = ["average_wait_time", "throughput_rate", "coordination_efficiency", "total_fuel_consumption"]
        titles = ["Average Waiting Time (s)", "Throughput Rate (veh/step)",
                 "Coordination Efficiency", "Total Fuel Consumption"]

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]

            model_val = model_result.get(metric, 0)
            baseline_val = baseline_result.get(metric, 0)

            values = [model_val, baseline_val]
            labels = ["Trained Model", "Baseline"]
            colors = ["skyblue", "lightcoral"]

            bars = ax.bar(labels, values, color=colors, alpha=0.7)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom')

            ax.set_title(title)
            ax.set_ylabel("Value")

            # Calculate improvement
            if baseline_val != 0:
                if "time" in metric or "consumption" in metric:
                    improvement = (baseline_val - model_val) / baseline_val * 100
                else:
                    improvement = (model_val - baseline_val) / baseline_val * 100

                ax.text(0.5, 0.95, f"Improvement: {improvement:+.1f}%",
                       transform=ax.transAxes, ha="center", va="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        plt.tight_layout()

        if save_plots:
            plot_path = output_path / f"performance_comparison.{plot_format}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance comparison plot to {plot_path}")

        plt.show()

    # Model comparison plot
    if "model_comparison" in results:
        comparison = results["model_comparison"]

        if len(comparison["models"]) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle("Multi-Model Performance Comparison", fontsize=16)

            # Metrics comparison
            metrics_comp = comparison["metrics_comparison"]
            models = comparison["models"]

            # Select key metrics for visualization
            key_metrics = ["average_wait_time", "throughput_rate"]
            metric_labels = ["Avg. Waiting Time (s)", "Throughput Rate"]

            for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
                if metric in metrics_comp:
                    ax = axes[i]
                    values = metrics_comp[metric]["values"]

                    bars = ax.bar(models, values, alpha=0.7, color=plt.cm.Set3(range(len(models))))

                    # Add value labels
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.2f}', ha='center', va='bottom')

                    ax.set_title(label)
                    ax.set_ylabel("Value")
                    ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            if save_plots:
                plot_path = output_path / f"model_comparison.{plot_format}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved model comparison plot to {plot_path}")

            plt.show()

    # Transfer learning visualization
    if "transfer_learning" in results:
        transfer_data = results["transfer_learning"]

        if "transfer_report" in transfer_data:
            transfer_report = transfer_data["transfer_report"]

            if "detailed_results" in transfer_report:
                detailed_results = transfer_report["detailed_results"]

                # Create transfer learning heatmap
                scenario_pairs = list(detailed_results.keys())
                retention_values = [detailed_results[pair]["transfer_learning_retention"]
                                  for pair in scenario_pairs]

                if scenario_pairs:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Create heatmap-like visualization
                    y_pos = np.arange(len(scenario_pairs))
                    colors = plt.cm.RdYlGn([val for val in retention_values])

                    bars = ax.barh(y_pos, retention_values, color=colors, alpha=0.7)

                    # Add value labels
                    for bar, val in zip(bars, retention_values):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2.,
                               f'{val:.3f}', ha='left', va='center')

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([pair.replace("_to_", " → ") for pair in scenario_pairs])
                    ax.set_xlabel("Transfer Learning Retention")
                    ax.set_title("Transfer Learning Performance")

                    # Add target line
                    target_retention = results.get("config", {}).get("target_metrics", {}).get("transfer_learning_retention", 0.7)
                    ax.axvline(x=target_retention, color='red', linestyle='--',
                              label=f'Target ({target_retention})')
                    ax.legend()

                    plt.tight_layout()

                    if save_plots:
                        plot_path = output_path / f"transfer_learning.{plot_format}"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Saved transfer learning plot to {plot_path}")

                    plt.show()

    logger.info("Visualization creation completed")


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    args: argparse.Namespace
) -> None:
    """Save evaluation results to files.

    Args:
        results: Evaluation results dictionary.
        output_dir: Output directory.
        args: Command line arguments.
    """
    logger.info(f"Saving results to {output_dir}")

    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save complete results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary report
    summary = create_summary_report(results, args)
    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed CSV files
    if "model_evaluation" in results and "episode_results" in results["model_evaluation"]:
        episode_data = results["model_evaluation"]["episode_results"]
        df = pd.DataFrame(episode_data)
        csv_file = output_path / "episode_results.csv"
        df.to_csv(csv_file, index=False)

    logger.info("Results saved successfully")


def create_summary_report(results: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Create evaluation summary report.

    Args:
        results: Evaluation results.
        args: Command line arguments.

    Returns:
        Summary report dictionary.
    """
    summary = {
        "evaluation_config": {
            "model_path": args.model_path,
            "scenario": args.scenario,
            "episodes": args.episodes,
            "deterministic": args.deterministic,
        },
        "performance_summary": {},
        "comparison_summary": {},
        "transfer_learning_summary": {}
    }

    # Performance summary
    if "model_evaluation" in results:
        model_result = results["model_evaluation"]
        summary["performance_summary"] = {
            "average_wait_time": model_result.get("average_wait_time", 0),
            "throughput_rate": model_result.get("throughput_rate", 0),
            "coordination_efficiency": model_result.get("coordination_efficiency", 0),
            "evaluation_time": model_result.get("evaluation_time", 0),
        }

    # Comparison summary
    if "model_comparison" in results:
        comparison = results["model_comparison"]
        summary["comparison_summary"] = {
            "models_compared": len(comparison["models"]),
            "best_overall": comparison["rankings"]["overall"][0],
            "significant_differences": {}
        }

        # Statistical significance
        if "statistical_tests" in comparison:
            for metric, test_result in comparison["statistical_tests"].items():
                summary["comparison_summary"]["significant_differences"][metric] = test_result.get("significant", False)

    # Transfer learning summary
    if "transfer_learning" in results:
        transfer_data = results["transfer_learning"]
        if "transfer_report" in transfer_data and "summary" in transfer_data["transfer_report"]:
            transfer_summary = transfer_data["transfer_report"]["summary"]
            summary["transfer_learning_summary"] = {
                "mean_transfer_score": transfer_summary.get("mean_transfer_score", 0),
                "mean_retention": transfer_summary.get("mean_retention", 0),
                "evaluations_conducted": transfer_summary.get("num_evaluations", 0),
            }

    return summary


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Set up logging
        log_level = args.log_level
        if args.verbose:
            log_level = "DEBUG"
        elif args.quiet:
            log_level = "WARNING"

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set random seed
        set_random_seeds(args.seed)

        logger.info("Starting evaluation...")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Episodes: {args.episodes}")

        # Load configuration
        config = load_config(args.config)
        if args.scenario:
            config.environment.scenario = args.scenario

        results = {"config": config.to_dict()}

        # Load main model
        model = load_model(args.model_path, config)

        # Evaluate main model
        model_results = evaluate_model(
            model, config, config.environment.scenario,
            args.episodes, args.deterministic
        )
        results["model_evaluation"] = model_results

        # Evaluate baseline if requested
        if args.baseline:
            baseline_results = evaluate_baseline(
                config, config.environment.scenario, args.baseline_episodes
            )
            results["baseline_evaluation"] = baseline_results

        # Model comparison
        if args.compare_with:
            logger.info("Loading comparison models...")
            comparison_results = [model_results]
            model_names = ["Main Model"]

            if args.model_names:
                model_names[0] = args.model_names[0]

            for i, compare_path in enumerate(args.compare_with):
                compare_model = load_model(compare_path, config)
                compare_results = evaluate_model(
                    compare_model, config, config.environment.scenario,
                    args.episodes, args.deterministic
                )
                comparison_results.append(compare_results)

                if args.model_names and i + 1 < len(args.model_names):
                    model_names.append(args.model_names[i + 1])
                else:
                    model_names.append(f"Model_{i + 2}")

            # Perform comparison
            comparison = compare_models(comparison_results, model_names, config)
            results["model_comparison"] = comparison

        # Transfer learning evaluation
        if args.transfer_learning:
            transfer_results = evaluate_transfer_learning(
                model, config, args.source_scenarios, args.target_scenarios
            )
            results["transfer_learning"] = transfer_results

        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        save_results(results, args.output_dir, args)

        # Generate visualizations
        if args.visualize:
            create_visualizations(results, args.output_dir, args.save_plots, args.plot_format)

        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")

        if "model_evaluation" in results:
            model_result = results["model_evaluation"]
            logger.info(f"Average waiting time: {model_result.get('average_wait_time', 0):.2f}s")
            logger.info(f"Throughput rate: {model_result.get('throughput_rate', 0):.3f} veh/step")
            logger.info(f"Coordination efficiency: {model_result.get('coordination_efficiency', 0):.3f}")

        if "baseline_evaluation" in results:
            baseline_result = results["baseline_evaluation"]
            model_wait = results["model_evaluation"].get('average_wait_time', 0)
            baseline_wait = baseline_result.get('average_wait_time', 0)

            if baseline_wait > 0:
                improvement = (baseline_wait - model_wait) / baseline_wait * 100
                logger.info(f"Waiting time improvement over baseline: {improvement:.1f}%")

        if "model_comparison" in results:
            comparison = results["model_comparison"]
            best_model = comparison["rankings"]["overall"][0]
            logger.info(f"Best performing model: {best_model}")

        if "transfer_learning" in results:
            transfer_data = results["transfer_learning"]
            if "transfer_report" in transfer_data and "summary" in transfer_data["transfer_report"]:
                transfer_summary = transfer_data["transfer_report"]["summary"]
                mean_retention = transfer_summary.get("mean_retention", 0)
                logger.info(f"Mean transfer learning retention: {mean_retention:.3f}")

        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("Evaluation completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())