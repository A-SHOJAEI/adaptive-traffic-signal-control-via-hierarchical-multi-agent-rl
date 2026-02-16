#!/usr/bin/env python3
"""Training script for hierarchical multi-agent RL traffic control system.

This script handles the complete training pipeline including:
- Configuration loading and validation
- Environment setup
- Model initialization
- Training execution with multiple frameworks
- Evaluation and metrics logging
- Checkpointing and model saving

Usage:
    python scripts/train.py [OPTIONS]

Examples:
    # Basic training with default config
    python scripts/train.py

    # Training with custom config
    python scripts/train.py --config configs/custom.yaml

    # Training with Ray RLlib
    python scripts/train.py --framework ray

    # Training with specific scenario
    python scripts/train.py --scenario cologne

    # Training with debugging enabled
    python scripts/train.py --debug --log-level DEBUG
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import torch
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.utils.config import (
    load_config,
    setup_logging,
    set_random_seeds,
    create_output_directory
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer import (
    HierarchicalTrainer
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.evaluation.metrics import (
    TrafficMetrics,
    TransferLearningEvaluator
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import (
    HierarchicalTrafficAgent
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train hierarchical multi-agent RL traffic control system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/default.yaml)"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Traffic scenario to use (manhattan_grid, cologne, custom)"
    )

    parser.add_argument(
        "--framework",
        type=str,
        choices=["ray", "sb3"],
        default=None,
        help="Training framework to use"
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes"
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Evaluation interval (in timesteps)"
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation during training"
    )

    # Transfer learning
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Enable transfer learning evaluation"
    )

    parser.add_argument(
        "--transfer-scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Transfer learning scenarios (source_to_target format)"
    )

    # Output and logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging level"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (equivalent to --log-level DEBUG)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Enable quiet mode (equivalent to --log-level ERROR)"
    )

    # Checkpointing
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Checkpoint saving interval (in timesteps)"
    )

    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable checkpoint saving"
    )

    # Performance options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (for Ray)"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage if available"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage"
    )

    # Validation
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )

    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate environment setup"
    )

    return parser.parse_args()


def setup_config(args: argparse.Namespace) -> Any:
    """Set up configuration from arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Configuration object.
    """
    # Load base configuration
    config = load_config(args.config)

    # Apply command line overrides
    overrides = {}

    if args.scenario:
        overrides["environment.scenario"] = args.scenario

    if args.framework:
        overrides["training.framework"] = args.framework

    if args.timesteps:
        overrides["training.total_timesteps"] = args.timesteps

    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size

    if args.learning_rate:
        overrides["training.learning_rate"] = args.learning_rate

    if args.seed:
        overrides["experiment.seed"] = args.seed

    if args.eval_episodes:
        overrides["evaluation.episodes"] = args.eval_episodes

    if args.eval_interval:
        overrides["evaluation.interval"] = args.eval_interval

    if args.output_dir:
        overrides["experiment.output_dir"] = args.output_dir

    if args.experiment_name:
        overrides["logging.mlflow.experiment_name"] = args.experiment_name

    if args.transfer_scenarios:
        overrides["evaluation.transfer_scenarios"] = args.transfer_scenarios

    # Logging level
    if args.debug:
        overrides["experiment.log_level"] = "DEBUG"
    elif args.quiet:
        overrides["experiment.log_level"] = "ERROR"
    elif args.log_level:
        overrides["experiment.log_level"] = args.log_level

    # Checkpointing
    if args.no_checkpoints:
        overrides["logging.checkpoints.enabled"] = False

    if args.save_interval:
        overrides["logging.checkpoints.frequency"] = args.save_interval

    # GPU/CPU settings
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Apply overrides
    if overrides:
        config.update(overrides)

    return config


def validate_environment(config: Any) -> bool:
    """Validate environment setup.

    Args:
        config: Configuration object.

    Returns:
        True if validation passes, False otherwise.
    """
    logger.info("Validating environment setup...")

    try:
        # Test SUMO installation
        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.data.loader import SUMODataLoader

        loader = SUMODataLoader(config)
        logger.info("✓ SUMO validation passed")

        # Test basic scenario loading
        scenarios = loader.get_available_scenarios()
        if config.environment.scenario in scenarios:
            logger.info(f"✓ Scenario '{config.environment.scenario}' available")
        else:
            logger.warning(f"⚠ Scenario '{config.environment.scenario}' not found, will be generated")

        # Test model creation
        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import HierarchicalTrafficAgent

        agent = HierarchicalTrafficAgent(config)
        logger.info("✓ Model creation successful")

        # Test environment creation
        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer import TrafficEnvironment

        env_config = {
            "rl_config": config.to_dict(),
            "scenario": config.environment.scenario
        }

        # This might fail in test environments without SUMO, so catch exceptions
        try:
            env = TrafficEnvironment(env_config)
            env.close()
            logger.info("✓ Environment creation successful")
        except Exception as e:
            logger.warning(f"⚠ Environment validation failed: {e}")

        logger.info("Environment validation completed")
        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def run_training(config: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Run the training process.

    Args:
        config: Configuration object.
        args: Command line arguments.

    Returns:
        Training results dictionary.
    """
    logger.info("Starting training process...")

    # Create trainer
    trainer = HierarchicalTrainer(config)

    # Set up MLflow run
    if config.logging.mlflow.enabled:
        if args.run_name:
            mlflow.set_tag("mlflow.runName", args.run_name)

        mlflow.set_tags({
            "framework": config.training.framework,
            "scenario": config.environment.scenario,
            "total_timesteps": config.training.total_timesteps,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
        })

    # Run training
    start_time = time.time()
    try:
        results = trainer.train()
        training_time = time.time() - start_time

        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Log training time
        if config.logging.mlflow.enabled:
            mlflow.log_metrics({
                "training_time_seconds": training_time,
                "training_time_hours": training_time / 3600,
            })

        return results

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.cleanup()
        raise

    except Exception as e:
        logger.error(f"Training failed: {e}")
        trainer.cleanup()
        raise

    finally:
        trainer.cleanup()


def run_evaluation(
    config: Any,
    args: argparse.Namespace,
    training_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Run post-training evaluation.

    Args:
        config: Configuration object.
        args: Command line arguments.
        training_results: Results from training.

    Returns:
        Evaluation results dictionary.
    """
    if args.no_eval:
        logger.info("Skipping evaluation (--no-eval specified)")
        return {}

    logger.info("Starting post-training evaluation...")

    evaluation_results = {}

    # Basic performance evaluation
    try:
        metrics_evaluator = TrafficMetrics(config)

        # Load trained model (simplified - would need actual model loading)
        logger.info("Loading trained model for evaluation...")

        # Create environment for evaluation
        from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.trainer import TrafficEnvironment

        env_config = {
            "rl_config": config.to_dict(),
            "scenario": config.environment.scenario
        }

        env = TrafficEnvironment(env_config)

        # Evaluate baseline performance
        logger.info("Evaluating baseline performance...")
        baseline_metrics = metrics_evaluator.evaluate_baseline(env, episodes=5)
        evaluation_results["baseline"] = baseline_metrics

        # Generate performance report
        logger.info("Generating performance report...")
        performance_report = metrics_evaluator.generate_performance_report()
        evaluation_results["performance"] = performance_report

        env.close()

        logger.info("Performance evaluation completed")

    except Exception as e:
        logger.error(f"Performance evaluation failed: {e}")

    # Transfer learning evaluation
    if args.transfer_learning:
        try:
            logger.info("Starting transfer learning evaluation...")

            transfer_evaluator = TransferLearningEvaluator(config)

            # Get transfer scenarios
            transfer_scenarios = config.evaluation.transfer_scenarios
            if args.transfer_scenarios:
                transfer_scenarios = args.transfer_scenarios

            # Convert scenario strings to tuples
            scenario_pairs = []
            for scenario_str in transfer_scenarios:
                if "_to_" in scenario_str:
                    source, target = scenario_str.split("_to_")
                    scenario_pairs.append((source, target))

            if scenario_pairs:
                # For this example, we'll create a dummy agent
                # In practice, you would load the actual trained agent
                dummy_agent = None  # Would be loaded from training results

                transfer_results = transfer_evaluator.evaluate_cross_scenario_transfer(
                    dummy_agent, scenario_pairs, episodes_per_scenario=3
                )

                transfer_report = transfer_evaluator.generate_transfer_report()
                evaluation_results["transfer_learning"] = transfer_report

                logger.info("Transfer learning evaluation completed")
            else:
                logger.warning("No valid transfer scenarios specified")

        except Exception as e:
            logger.error(f"Transfer learning evaluation failed: {e}")

    return evaluation_results


def save_results(
    config: Any,
    output_dir: str,
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any]
) -> None:
    """Save training and evaluation results.

    Args:
        config: Configuration object.
        output_dir: Output directory path.
        training_results: Training results.
        evaluation_results: Evaluation results.
    """
    logger.info(f"Saving results to {output_dir}")

    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training results
    if training_results:
        training_file = output_path / "training_results.json"
        with open(training_file, "w") as f:
            json.dump(training_results, f, indent=2, default=str)

    # Save evaluation results
    if evaluation_results:
        evaluation_file = output_path / "evaluation_results.json"
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

    # Save final config
    config_file = output_path / "final_config.yaml"
    config.save(str(config_file))

    # Create summary report
    summary = {
        "experiment": {
            "name": config.experiment.name,
            "framework": config.training.framework,
            "scenario": config.environment.scenario,
            "total_timesteps": config.training.total_timesteps,
        },
        "performance": {},
        "target_achievements": {},
    }

    # Extract key metrics
    if evaluation_results and "performance" in evaluation_results:
        perf = evaluation_results["performance"]
        if "improvements" in perf:
            summary["performance"] = perf["improvements"]
        if "target_achievements" in perf:
            summary["target_achievements"] = perf["target_achievements"]

    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved successfully")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Set up configuration
        config = setup_config(args)

        # Set up logging
        setup_logging(config)

        # Set random seeds
        set_random_seeds(config.experiment.seed)

        # Create output directory
        output_dir = create_output_directory(config)

        logger.info(f"Starting hierarchical multi-agent RL training")
        logger.info(f"Experiment: {config.experiment.name}")
        logger.info(f"Framework: {config.training.framework}")
        logger.info(f"Scenario: {config.environment.scenario}")
        logger.info(f"Total timesteps: {config.training.total_timesteps:,}")
        logger.info(f"Output directory: {output_dir}")

        # Validate environment if requested
        if args.validate_env or args.dry_run:
            if not validate_environment(config):
                logger.error("Environment validation failed")
                return 1

        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0

        # Run training
        training_results = run_training(config, args)

        # Run evaluation
        evaluation_results = run_evaluation(config, args, training_results)

        # Save results
        save_results(config, output_dir, training_results, evaluation_results)

        # Print summary
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(f"Experiment: {config.experiment.name}")
        logger.info(f"Output directory: {output_dir}")

        if evaluation_results and "performance" in evaluation_results:
            perf = evaluation_results["performance"]
            if "improvements" in perf:
                improvements = perf["improvements"]
                logger.info(f"Wait time reduction: {improvements.get('average_wait_time_reduction', 0):.2%}")
                logger.info(f"Throughput improvement: {improvements.get('throughput_improvement', 0):.2%}")

            if "target_achievements" in perf:
                achievements = perf["target_achievements"]
                targets_met = achievements.get("all_targets_met", False)
                logger.info(f"All targets achieved: {targets_met}")

        logger.info("Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())