#!/usr/bin/env python3
"""Prediction script for trained hierarchical traffic control models."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.models.model import (
    HierarchicalTrafficAgent,
    TrafficEnvironmentWrapper
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.synthetic_env import (
    SyntheticTrafficEnvironment
)
from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str, config_path: str) -> HierarchicalTrafficAgent:
    """Load a trained hierarchical traffic control model.

    Args:
        model_path: Path to saved model checkpoint.
        config_path: Path to configuration file.

    Returns:
        Loaded hierarchical traffic agent.
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    logger.info("Initializing hierarchical traffic agent")
    agent = HierarchicalTrafficAgent(config)

    # Add agents based on grid size
    grid_size = config.environment.grid_size
    num_intersections = grid_size[0] * grid_size[1]

    # Add intersection agents
    for i in range(num_intersections):
        agent.add_intersection_agent(f"intersection_{i}")

    # Add district agents
    district_size = config.agents.high_level.district_size
    num_districts = (grid_size[0] // district_size[0]) * (grid_size[1] // district_size[1])
    for i in range(num_districts):
        agent.add_district_agent(f"district_{i}")

    if os.path.exists(model_path):
        logger.info(f"Loading model weights from {model_path}")
        agent.load_checkpoint(model_path)
    else:
        logger.warning(f"Model checkpoint not found at {model_path}. Using randomly initialized model.")

    return agent


def generate_sample_observation(env: SyntheticTrafficEnvironment, agent_id: str) -> np.ndarray:
    """Generate a sample observation for prediction.

    Args:
        env: Traffic environment.
        agent_id: Agent identifier.

    Returns:
        Sample observation array.
    """
    # Reset environment to get initial observation
    obs, _ = env.reset()
    return obs.get(agent_id, np.zeros(64))


def predict_actions(
    agent: HierarchicalTrafficAgent,
    env: SyntheticTrafficEnvironment,
    deterministic: bool = True,
    num_steps: int = 10
) -> Dict:
    """Run predictions on the environment.

    Args:
        agent: Trained hierarchical traffic agent.
        env: Traffic environment.
        deterministic: Whether to use deterministic actions.
        num_steps: Number of prediction steps to run.

    Returns:
        Dictionary containing prediction results and statistics.
    """
    logger.info(f"Running predictions for {num_steps} steps")

    wrapper = TrafficEnvironmentWrapper(env.config, agent)

    observations, _ = env.reset()
    episode_rewards = []
    all_actions = {}
    step_info = []

    for step in range(num_steps):
        # Predict actions
        actions = wrapper.predict(observations, deterministic=deterministic)

        # Store actions
        for agent_id, action in actions.items():
            if agent_id not in all_actions:
                all_actions[agent_id] = []
            all_actions[agent_id].append(action.tolist() if isinstance(action, np.ndarray) else action)

        # Step environment
        observations, rewards, dones, truncated, infos = env.step(actions)

        # Calculate step statistics
        step_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
        episode_rewards.append(step_reward)

        step_info.append({
            "step": step,
            "total_reward": step_reward,
            "num_agents": len(actions),
            "actions": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in actions.items()}
        })

        if dones.get('__all__', False):
            logger.info(f"Episode completed at step {step}")
            break

    # Calculate statistics
    results = {
        "num_steps": len(episode_rewards),
        "total_reward": sum(episode_rewards),
        "mean_reward_per_step": np.mean(episode_rewards),
        "std_reward_per_step": np.std(episode_rewards),
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "actions_summary": {
            agent_id: {
                "num_actions": len(actions),
                "unique_actions": len(set(map(str, actions)))
            }
            for agent_id, actions in all_actions.items()
        },
        "detailed_steps": step_info[:5]  # First 5 steps for brevity
    }

    return results


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(
        description="Run predictions using trained hierarchical traffic control model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/hierarchical_marl_traffic_20260208_073112/checkpoint_best.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file with observations (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Path to save prediction results"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions instead of stochastic"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of prediction steps to run"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize predictions (if supported)"
    )

    args = parser.parse_args()

    # Load model
    agent = load_trained_model(args.model_path, args.config_path)

    # Create environment
    logger.info("Creating synthetic traffic environment")
    config = load_config(args.config_path)
    env = SyntheticTrafficEnvironment(config)

    # Run predictions
    if args.input:
        logger.info(f"Loading input observations from {args.input}")
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        # Custom input processing would go here
        results = predict_actions(agent, env, args.deterministic, args.num_steps)
    else:
        logger.info("Generating predictions on synthetic environment")
        results = predict_actions(agent, env, args.deterministic, args.num_steps)

    # Print results
    logger.info("\nPrediction Results:")
    logger.info(f"  Total steps: {results['num_steps']}")
    logger.info(f"  Total reward: {results['total_reward']:.2f}")
    logger.info(f"  Mean reward per step: {results['mean_reward_per_step']:.2f} +/- {results['std_reward_per_step']:.2f}")
    logger.info(f"  Reward range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    logger.info(f"  Number of agents: {len(results['actions_summary'])}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nPrediction results saved to {output_path}")

    if args.visualize:
        logger.info("Visualization requested but not yet implemented")

    return 0


if __name__ == "__main__":
    sys.exit(main())
