#!/usr/bin/env python3
"""Training script using synthetic traffic environment (no SUMO dependency).

This script trains the hierarchical multi-agent RL system using a pure-Python
synthetic traffic grid environment. It implements:
  1. Pre-training of low-level intersection agents (PPO)
  2. Training of high-level district coordination agents (PPO with continuous actions)
  3. Joint hierarchical fine-tuning
  4. Evaluation and metrics reporting

Usage:
    python scripts/train_synthetic.py
    python scripts/train_synthetic.py --timesteps 50000
    python scripts/train_synthetic.py --grid-rows 3 --grid-cols 3
"""

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from adaptive_traffic_signal_control_via_hierarchical_multi_agent_rl.training.synthetic_env import (
    SyntheticTrafficEnvironment,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_synthetic")


# ---------------------------------------------------------------------------
# Utility: set seeds
# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# PPO Networks
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    """Actor-Critic network for PPO with discrete or continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        continuous: bool = False,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.continuous = continuous
        self.use_lstm = use_lstm

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.features = nn.Sequential(*layers)

        # Optional LSTM
        if use_lstm:
            self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
            feature_out = lstm_hidden
        else:
            self.lstm = None
            feature_out = in_dim

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(feature_out, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(feature_out, action_dim)

        # Critic head
        self.critic = nn.Linear(feature_out, 1)

        # Weight init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(
        self, obs: torch.Tensor, lstm_state: Optional[Tuple] = None
    ) -> Tuple:
        features = self.features(obs)

        if self.lstm is not None:
            if features.dim() == 2:
                features = features.unsqueeze(1)
            features, lstm_state = self.lstm(features, lstm_state)
            features = features.squeeze(1)

        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            return mean, std, value, lstm_state
        else:
            logits = self.actor(features)
            return logits, value, lstm_state


# ---------------------------------------------------------------------------
# PPO Rollout Buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return returns, advantages

    def get_batches(
        self, batch_size: int, returns: np.ndarray, advantages: np.ndarray
    ):
        n = len(self.observations)
        indices = np.arange(n)
        np.random.shuffle(indices)

        obs_arr = np.array(self.observations, dtype=np.float32)
        act_arr = np.array(self.actions)
        lp_arr = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield (
                obs_arr[idx],
                act_arr[idx],
                lp_arr[idx],
                returns[idx],
                advantages[idx],
            )

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


# ---------------------------------------------------------------------------
# PPO Update
# ---------------------------------------------------------------------------
def ppo_update(
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 64,
    clip_range: float = 0.2,
    entropy_coeff: float = 0.01,
    vf_coeff: float = 0.5,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> Dict[str, float]:
    """Perform PPO update on the model."""
    returns, advantages = buffer.compute_returns_and_advantages(
        gamma, gae_lambda, last_value
    )
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(epochs):
        for obs_b, act_b, old_lp_b, ret_b, adv_b in buffer.get_batches(
            batch_size, returns, advantages
        ):
            obs_t = torch.FloatTensor(obs_b).to(device)
            ret_t = torch.FloatTensor(ret_b).to(device)
            adv_t = torch.FloatTensor(adv_b).to(device)
            old_lp_t = torch.FloatTensor(old_lp_b).to(device)

            if model.continuous:
                mean, std, values, _ = model(obs_t)
                dist = Normal(mean, std)
                act_t = torch.FloatTensor(act_b).to(device)
                new_log_probs = dist.log_prob(act_t).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            else:
                logits, values, _ = model(obs_t)
                dist = Categorical(logits=logits)
                act_t = torch.LongTensor(act_b).to(device)
                new_log_probs = dist.log_prob(act_t)
                entropy = dist.entropy().mean()

            values = values.squeeze(-1)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, ret_t)

            # Total loss
            loss = policy_loss + vf_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    buffer.clear()

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


# ---------------------------------------------------------------------------
# Communication Network for Hierarchical Coordination
# ---------------------------------------------------------------------------
class CommunicationNet(nn.Module):
    """Simple communication network between intersection and district agents."""

    def __init__(self, intersection_feature_dim: int, comm_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(intersection_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(comm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, intersection_feature_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        message = self.encoder(x)
        reconstructed = self.decoder(message)
        return message, reconstructed


# ---------------------------------------------------------------------------
# Phase 1: Pre-train intersection agents (multi-agent mode)
# ---------------------------------------------------------------------------
def pretrain_intersection_agents(
    env: SyntheticTrafficEnvironment,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[PPOActorCritic, Dict[str, Any]]:
    """Pre-train low-level intersection agents using PPO in multi-agent mode.

    All intersection agents share a single policy network (parameter sharing).
    Experience is collected from all intersections simultaneously, giving the
    shared policy a diverse set of training examples each step.

    Args:
        env: Synthetic traffic environment.
        config: Training configuration.
        device: Torch device.

    Returns:
        Tuple of (trained model, training metrics).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Pre-training intersection agents (PPO, multi-agent)")
    logger.info("=" * 60)

    obs_dim = config["obs_dim"]
    action_dim = config["num_phases"]
    hidden_dims = config.get("low_level_hidden", [256, 256, 128])
    lr = config.get("learning_rate", 3e-4)
    total_steps = config.get("pretrain_steps", 50000)
    rollout_length = config.get("rollout_length", 512)
    batch_size = config.get("batch_size", 256)
    gamma = config.get("gamma", 0.99)
    gae_lambda = config.get("gae_lambda", 0.95)
    clip_range = config.get("clip_range", 0.2)
    epochs_per_update = config.get("ppo_epochs", 4)

    model = PPOActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        continuous=False,
        use_lstm=False,  # Simpler model for multi-agent shared policy
        lstm_hidden=config.get("lstm_hidden", 64),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer()

    obs, info = env.reset()
    env_steps_done = 0  # environment steps (each = 1 call to step_multi_agent)
    total_experience = 0  # total individual agent transitions collected
    episode_rewards = []
    current_episode_reward = 0.0
    episode_count = 0
    best_mean_reward = float("-inf")

    metrics_history = {
        "episode_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
    }

    start_time = time.time()
    num_intersections = len(env.intersection_ids)

    while total_experience < total_steps:
        # Collect rollout: each env step generates num_intersections transitions
        for _ in range(rollout_length):
            # Get observations for all intersections
            int_obs_list = []
            for i in range(num_intersections):
                int_obs_list.append(env._get_observation(i))
            int_obs_batch = torch.FloatTensor(np.array(int_obs_list)).to(device)

            with torch.no_grad():
                logits, values, _ = model(int_obs_batch)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Build action dict
            action_dict = {}
            for i, iid in enumerate(env.intersection_ids):
                action_dict[iid] = actions[i].item()
            # District agents: no-op (zeros) during Phase 1
            for did in env.district_ids:
                action_dict[did] = np.zeros(env.high_level_action_dim)

            # Step environment
            observations, rewards, terminateds, truncateds, infos = env.step_multi_agent(action_dict)

            done = terminateds.get("__all__", False)

            # Add each intersection's experience to the shared buffer
            for i, iid in enumerate(env.intersection_ids):
                buffer.add(
                    int_obs_list[i],
                    actions[i].item(),
                    rewards.get(iid, 0.0),
                    done,
                    log_probs[i].item(),
                    values[i].item(),
                )
                total_experience += 1

            # Track episode reward (sum across all intersection agents)
            step_reward = sum(rewards.get(iid, 0.0) for iid in env.intersection_ids)
            current_episode_reward += step_reward
            env_steps_done += 1

            if done:
                episode_rewards.append(current_episode_reward)
                metrics_history["episode_rewards"].append(current_episode_reward)
                episode_count += 1
                current_episode_reward = 0.0
                obs, info = env.reset()

            if total_experience >= total_steps:
                break

        # Get last value for GAE (average over all intersections)
        with torch.no_grad():
            int_obs_list_last = []
            for i in range(num_intersections):
                int_obs_list_last.append(env._get_observation(i))
            last_obs_t = torch.FloatTensor(np.array(int_obs_list_last)).to(device)
            _, last_vals, _ = model(last_obs_t)
            last_value = last_vals.mean().item()

        # PPO update
        update_metrics = ppo_update(
            model, optimizer, buffer, device,
            epochs=epochs_per_update,
            batch_size=batch_size,
            clip_range=clip_range,
            gamma=gamma,
            gae_lambda=gae_lambda,
            last_value=last_value,
        )

        metrics_history["policy_losses"].append(update_metrics["policy_loss"])
        metrics_history["value_losses"].append(update_metrics["value_loss"])
        metrics_history["entropies"].append(update_metrics["entropy"])

        # Logging
        if episode_rewards:
            recent_rewards = episode_rewards[-20:]
            mean_reward = np.mean(recent_rewards)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

            if total_experience % 10000 < rollout_length * num_intersections:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Exp {total_experience:>7d}/{total_steps} | "
                    f"EnvSteps: {env_steps_done:>5d} | "
                    f"Episodes: {episode_count:>4d} | "
                    f"Mean Reward (20ep): {mean_reward:>8.3f} | "
                    f"Best: {best_mean_reward:>8.3f} | "
                    f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                    f"Value Loss: {update_metrics['value_loss']:.4f} | "
                    f"Entropy: {update_metrics['entropy']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

    elapsed = time.time() - start_time
    final_mean = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0

    logger.info(f"Phase 1 complete: {episode_count} episodes, "
                f"mean reward (last 20): {final_mean:.3f}, "
                f"best mean reward: {best_mean_reward:.3f}, "
                f"time: {elapsed:.1f}s")

    results = {
        "phase": "pretrain_intersection",
        "total_experience": total_experience,
        "env_steps": env_steps_done,
        "total_episodes": episode_count,
        "final_mean_reward": float(final_mean),
        "best_mean_reward": float(best_mean_reward),
        "training_time_seconds": elapsed,
        "final_policy_loss": float(metrics_history["policy_losses"][-1]) if metrics_history["policy_losses"] else 0.0,
        "final_value_loss": float(metrics_history["value_losses"][-1]) if metrics_history["value_losses"] else 0.0,
        "final_entropy": float(metrics_history["entropies"][-1]) if metrics_history["entropies"] else 0.0,
    }

    return model, results


# ---------------------------------------------------------------------------
# Phase 2: Train district coordination agents
# ---------------------------------------------------------------------------
def train_district_agents(
    env: SyntheticTrafficEnvironment,
    intersection_model: PPOActorCritic,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[PPOActorCritic, Dict[str, Any]]:
    """Train high-level district coordination agents.

    Args:
        env: Synthetic traffic environment.
        intersection_model: Pre-trained intersection model.
        config: Training configuration.
        device: Torch device.

    Returns:
        Tuple of (district model, training metrics).
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Training district coordination agents (PPO-Continuous)")
    logger.info("=" * 60)

    high_obs_dim = config["high_level_obs_dim"]
    high_action_dim = config["high_level_action_dim"]
    hidden_dims = config.get("high_level_hidden", [512, 512, 256])
    lr = config.get("learning_rate", 3e-4)
    total_steps = config.get("district_steps", 30000)
    rollout_length = config.get("rollout_length", 1024)
    batch_size = config.get("batch_size", 64)

    district_model = PPOActorCritic(
        obs_dim=high_obs_dim,
        action_dim=high_action_dim,
        hidden_dims=hidden_dims,
        continuous=True,
    ).to(device)

    optimizer = torch.optim.Adam(district_model.parameters(), lr=lr)
    buffer = RolloutBuffer()

    # We train district agents in a multi-agent loop
    obs, info = env.reset()
    steps_done = 0
    episode_count = 0
    episode_rewards = []
    current_episode_reward = 0.0
    best_mean_reward = float("-inf")

    start_time = time.time()

    intersection_model.eval()

    while steps_done < total_steps:
        for _ in range(rollout_length):
            # Build multi-agent actions
            action_dict = {}

            # Intersection agents use pre-trained model
            for i, iid in enumerate(env.intersection_ids):
                int_obs = env._get_observation(i)
                int_obs_t = torch.FloatTensor(int_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, _ = intersection_model(int_obs_t)
                    dist_i = Categorical(logits=logits)
                    action_dict[iid] = dist_i.sample().item()

            # District agents use the district model
            district_obs_list = []
            for did in env.district_ids:
                d_obs = env._get_district_observation(did)
                district_obs_list.append(d_obs)

            if district_obs_list:
                # Use first district for training signal (simplified)
                d_obs = district_obs_list[0]
                d_obs_t = torch.FloatTensor(d_obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    mean, std, value, _ = district_model(d_obs_t)
                    dist_d = Normal(mean, std)
                    d_action = dist_d.sample()
                    d_log_prob = dist_d.log_prob(d_action).sum(dim=-1)

                d_action_np = d_action.squeeze(0).cpu().numpy()

                for j, did in enumerate(env.district_ids):
                    action_dict[did] = d_action_np

                buffer.add(
                    d_obs,
                    d_action_np,
                    0.0,  # reward filled after step
                    False,
                    d_log_prob.item(),
                    value.item(),
                )

            # Step environment
            observations, rewards, terminateds, truncateds, infos = env.step_multi_agent(action_dict)

            # Update buffer reward for district
            if buffer.rewards:
                district_reward = np.mean([
                    rewards.get(did, 0.0) for did in env.district_ids
                ])
                buffer.rewards[-1] = district_reward
                current_episode_reward += district_reward

            done = terminateds.get("__all__", False)
            if buffer.dones:
                buffer.dones[-1] = done

            steps_done += 1

            if done:
                episode_rewards.append(current_episode_reward)
                episode_count += 1
                current_episode_reward = 0.0
                obs, info = env.reset()

            if steps_done >= total_steps:
                break

        # PPO update for district model
        if len(buffer.observations) > 0:
            with torch.no_grad():
                last_d_obs = district_obs_list[0] if district_obs_list else np.zeros(high_obs_dim)
                last_d_t = torch.FloatTensor(last_d_obs).unsqueeze(0).to(device)
                _, _, last_val, _ = district_model(last_d_t)
                last_value = last_val.item()

            update_metrics = ppo_update(
                district_model, optimizer, buffer, device,
                epochs=4,
                batch_size=batch_size,
                clip_range=0.2,
                last_value=last_value,
            )

            if episode_rewards and steps_done % 5000 < rollout_length:
                recent = episode_rewards[-10:]
                mean_r = np.mean(recent)
                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                logger.info(
                    f"  Step {steps_done:>7d}/{total_steps} | "
                    f"Episodes: {episode_count:>4d} | "
                    f"Mean District Reward (10ep): {mean_r:>8.3f} | "
                    f"Best: {best_mean_reward:>8.3f} | "
                    f"Policy Loss: {update_metrics['policy_loss']:.4f}"
                )

    elapsed = time.time() - start_time
    final_mean = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0

    logger.info(f"Phase 2 complete: {episode_count} episodes, "
                f"mean reward (last 10): {final_mean:.3f}, "
                f"time: {elapsed:.1f}s")

    results = {
        "phase": "train_district",
        "total_steps": steps_done,
        "total_episodes": episode_count,
        "final_mean_reward": float(final_mean),
        "best_mean_reward": float(best_mean_reward),
        "training_time_seconds": elapsed,
    }

    return district_model, results


# ---------------------------------------------------------------------------
# Phase 3: Joint hierarchical fine-tuning
# ---------------------------------------------------------------------------
def joint_fine_tuning(
    env: SyntheticTrafficEnvironment,
    intersection_model: PPOActorCritic,
    district_model: PPOActorCritic,
    comm_net: CommunicationNet,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Joint fine-tuning of both agent levels with communication.

    Args:
        env: Synthetic traffic environment.
        intersection_model: Pre-trained intersection model.
        district_model: Trained district model.
        comm_net: Communication network.
        config: Training configuration.
        device: Torch device.

    Returns:
        Fine-tuning metrics.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Joint hierarchical fine-tuning with communication")
    logger.info("=" * 60)

    total_steps = config.get("joint_steps", 20000)
    lr = config.get("learning_rate", 1e-4)  # Lower LR for fine-tuning
    rollout_length = config.get("rollout_length", 512)
    batch_size = config.get("batch_size", 64)

    # Both models in training mode
    intersection_model.train()
    district_model.train()

    # Combined optimizer
    all_params = list(intersection_model.parameters()) + \
                 list(district_model.parameters()) + \
                 list(comm_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    intersection_buffer = RolloutBuffer()

    obs, info = env.reset()
    steps_done = 0
    episode_count = 0
    episode_rewards = []
    current_episode_reward = 0.0
    best_mean_reward = float("-inf")

    start_time = time.time()

    while steps_done < total_steps:
        for _ in range(rollout_length):
            action_dict = {}

            # Intersection agents with communication
            int_obs_list = []
            for i, iid in enumerate(env.intersection_ids):
                int_obs = env._get_observation(i)
                int_obs_list.append(int_obs)

            int_obs_batch = torch.FloatTensor(np.array(int_obs_list)).to(device)

            with torch.no_grad():
                # Generate communication messages
                messages, _ = comm_net(int_obs_batch)

                # Get intersection actions
                logits, values, _ = intersection_model(int_obs_batch)
                dist_i = Categorical(logits=logits)
                actions_i = dist_i.sample()
                log_probs_i = dist_i.log_prob(actions_i)

            for i, iid in enumerate(env.intersection_ids):
                action_dict[iid] = actions_i[i].item()

            # Store first intersection for buffer (simplified)
            intersection_buffer.add(
                int_obs_list[0],
                actions_i[0].item(),
                0.0,
                False,
                log_probs_i[0].item(),
                values[0].item(),
            )

            # District agents
            for did in env.district_ids:
                d_obs = env._get_district_observation(did)
                d_obs_t = torch.FloatTensor(d_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    mean, std, _, _ = district_model(d_obs_t)
                    d_dist = Normal(mean, std)
                    d_action = d_dist.sample()
                action_dict[did] = d_action.squeeze(0).cpu().numpy()

            # Step
            observations, rewards, terminateds, truncateds, infos = env.step_multi_agent(action_dict)

            # Compute combined reward
            combined_reward = np.mean([
                rewards.get(iid, 0.0) for iid in env.intersection_ids
            ])
            current_episode_reward += combined_reward

            if intersection_buffer.rewards:
                intersection_buffer.rewards[-1] = combined_reward

            done = terminateds.get("__all__", False)
            if intersection_buffer.dones:
                intersection_buffer.dones[-1] = done

            steps_done += 1

            if done:
                episode_rewards.append(current_episode_reward)
                episode_count += 1
                current_episode_reward = 0.0
                obs, info = env.reset()

            if steps_done >= total_steps:
                break

        # Update intersection model (fine-tune)
        if len(intersection_buffer.observations) > 0:
            with torch.no_grad():
                last_obs_t = torch.FloatTensor(int_obs_list[0]).unsqueeze(0).to(device)
                _, last_val, _ = intersection_model(last_obs_t)
                last_value = last_val.item()

            update_metrics = ppo_update(
                intersection_model, optimizer, intersection_buffer, device,
                epochs=2,
                batch_size=batch_size,
                clip_range=0.1,  # Smaller clip for fine-tuning
                last_value=last_value,
            )

            if episode_rewards and steps_done % 5000 < rollout_length:
                recent = episode_rewards[-10:]
                mean_r = np.mean(recent)
                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                logger.info(
                    f"  Step {steps_done:>7d}/{total_steps} | "
                    f"Episodes: {episode_count:>4d} | "
                    f"Mean Joint Reward (10ep): {mean_r:>8.3f} | "
                    f"Best: {best_mean_reward:>8.3f} | "
                    f"Policy Loss: {update_metrics['policy_loss']:.4f}"
                )

    elapsed = time.time() - start_time
    final_mean = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0

    logger.info(f"Phase 3 complete: {episode_count} episodes, "
                f"mean reward (last 10): {final_mean:.3f}, "
                f"time: {elapsed:.1f}s")

    return {
        "phase": "joint_fine_tuning",
        "total_steps": steps_done,
        "total_episodes": episode_count,
        "final_mean_reward": float(final_mean),
        "best_mean_reward": float(best_mean_reward),
        "training_time_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    env: SyntheticTrafficEnvironment,
    intersection_model: PPOActorCritic,
    district_model: PPOActorCritic,
    device: torch.device,
    num_episodes: int = 10,
    label: str = "Evaluation",
) -> Dict[str, Any]:
    """Evaluate the trained hierarchical system.

    Args:
        env: Traffic environment.
        intersection_model: Trained intersection model.
        district_model: Trained district model.
        device: Torch device.
        num_episodes: Number of evaluation episodes.
        label: Label for logging.

    Returns:
        Evaluation metrics.
    """
    logger.info(f"--- {label}: running {num_episodes} episodes ---")

    intersection_model.eval()
    district_model.eval()

    episode_rewards = []
    episode_lengths = []
    episode_waiting_times = []
    episode_throughputs = []
    episode_queue_totals = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_length = 0

        done = False
        while not done:
            action_dict = {}

            # Intersection actions (batch all intersections)
            int_obs_list = []
            for i in range(len(env.intersection_ids)):
                int_obs_list.append(env._get_observation(i))
            int_obs_batch = torch.FloatTensor(np.array(int_obs_list)).to(device)

            with torch.no_grad():
                logits, _, _ = intersection_model(int_obs_batch)
                # Use softmax sampling for evaluation (deterministic argmax can collapse)
                dist_eval = Categorical(logits=logits)
                actions_eval = dist_eval.sample()

            for i, iid in enumerate(env.intersection_ids):
                action_dict[iid] = actions_eval[i].item()

            # District actions
            for did in env.district_ids:
                d_obs = env._get_district_observation(did)
                d_obs_t = torch.FloatTensor(d_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    mean, _, _, _ = district_model(d_obs_t)
                action_dict[did] = mean.squeeze(0).cpu().numpy()

            observations, rewards, terminateds, truncateds, infos = env.step_multi_agent(action_dict)

            ep_reward += sum(r for k, r in rewards.items() if k != "__all__")
            ep_length += 1
            done = terminateds.get("__all__", False)

        # Collect episode info
        final_info = env._get_info()
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_waiting_times.append(final_info.get("episode_total_waiting", 0.0))
        episode_throughputs.append(final_info.get("episode_total_throughput", 0.0))
        episode_queue_totals.append(final_info.get("total_queue", 0.0))

    results = {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_total_waiting_time": float(np.mean(episode_waiting_times)),
        "mean_total_throughput": float(np.mean(episode_throughputs)),
        "mean_final_queue": float(np.mean(episode_queue_totals)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }

    logger.info(
        f"  {label} Results: "
        f"Mean Reward: {results['mean_reward']:.3f} +/- {results['std_reward']:.3f} | "
        f"Mean Waiting: {results['mean_total_waiting_time']:.1f} | "
        f"Mean Throughput: {results['mean_total_throughput']:.1f} | "
        f"Mean Final Queue: {results['mean_final_queue']:.1f}"
    )

    return results


# ---------------------------------------------------------------------------
# Baseline evaluation (random / fixed-time)
# ---------------------------------------------------------------------------
def evaluate_baseline(
    env: SyntheticTrafficEnvironment,
    num_episodes: int = 10,
    strategy: str = "fixed",
) -> Dict[str, Any]:
    """Evaluate a baseline strategy.

    Args:
        env: Traffic environment.
        num_episodes: Number of episodes.
        strategy: "fixed" (keep phase 0) or "random".

    Returns:
        Baseline metrics.
    """
    logger.info(f"--- Baseline ({strategy}): running {num_episodes} episodes ---")

    episode_rewards = []
    episode_waiting_times = []
    episode_throughputs = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action_dict = {}
            for iid in env.intersection_ids:
                if strategy == "random":
                    action_dict[iid] = np.random.randint(0, env.num_phases)
                else:
                    action_dict[iid] = 0
            for did in env.district_ids:
                action_dict[did] = np.zeros(env.high_level_action_dim)

            _, rewards, terminateds, _, _ = env.step_multi_agent(action_dict)
            ep_reward += sum(r for k, r in rewards.items() if k != "__all__")
            done = terminateds.get("__all__", False)

        final_info = env._get_info()
        episode_rewards.append(ep_reward)
        episode_waiting_times.append(final_info.get("episode_total_waiting", 0.0))
        episode_throughputs.append(final_info.get("episode_total_throughput", 0.0))

    results = {
        "strategy": strategy,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_total_waiting_time": float(np.mean(episode_waiting_times)),
        "mean_total_throughput": float(np.mean(episode_throughputs)),
    }

    logger.info(
        f"  Baseline ({strategy}): "
        f"Mean Reward: {results['mean_reward']:.3f} +/- {results['std_reward']:.3f} | "
        f"Mean Waiting: {results['mean_total_waiting_time']:.1f}"
    )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hierarchical multi-agent RL for traffic control (synthetic env)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--pretrain-steps", type=int, default=50000)
    parser.add_argument("--district-steps", type=int, default=30000)
    parser.add_argument("--joint-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    logger.info(f"Using device: {device}")

    # Environment config
    env_config = {
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "max_steps": args.max_steps,
        "num_phases": 4,
        "num_lanes_per_approach": 3,
        "base_arrival_rate": 0.3,
        "obs_dim": 64,
        "high_level_obs_dim": 128,
        "high_level_action_dim": 16,
        "district_size": (3, 3),
        "coordination_frequency": 10,
        "seed": args.seed,
    }

    # Training config
    train_config = {
        **env_config,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "pretrain_steps": args.pretrain_steps,
        "district_steps": args.district_steps,
        "joint_steps": args.joint_steps,
        "rollout_length": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ppo_epochs": 4,
        "low_level_hidden": [256, 256, 128],
        "high_level_hidden": [512, 512, 256],
        "use_lstm": True,
        "lstm_hidden": 64,
    }

    # Create environment
    env = SyntheticTrafficEnvironment(env_config)
    logger.info(f"Created environment: {args.grid_rows}x{args.grid_cols} grid, "
                f"{env.num_intersections} intersections, {env.num_districts} districts")

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"hierarchical_marl_traffic_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_start = time.time()

    # -----------------------------------------------------------------------
    # Baseline evaluation
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 60)
    baseline_fixed = evaluate_baseline(env, num_episodes=args.eval_episodes, strategy="fixed")
    baseline_random = evaluate_baseline(env, num_episodes=args.eval_episodes, strategy="random")
    all_results["baseline_fixed"] = baseline_fixed
    all_results["baseline_random"] = baseline_random

    # -----------------------------------------------------------------------
    # Phase 1: Pre-train intersection agents
    # -----------------------------------------------------------------------
    logger.info("")
    intersection_model, pretrain_results = pretrain_intersection_agents(
        env, train_config, device
    )
    all_results["phase1_pretrain"] = pretrain_results

    # -----------------------------------------------------------------------
    # Phase 2: Train district agents
    # -----------------------------------------------------------------------
    logger.info("")
    district_model, district_results = train_district_agents(
        env, intersection_model, train_config, device
    )
    all_results["phase2_district"] = district_results

    # -----------------------------------------------------------------------
    # Phase 3: Joint fine-tuning
    # -----------------------------------------------------------------------
    logger.info("")
    comm_net = CommunicationNet(
        intersection_feature_dim=env_config["obs_dim"],
        comm_dim=32,
    ).to(device)

    joint_results = joint_fine_tuning(
        env, intersection_model, district_model, comm_net, train_config, device
    )
    all_results["phase3_joint"] = joint_results

    # -----------------------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    eval_results = evaluate(
        env, intersection_model, district_model, device,
        num_episodes=args.eval_episodes,
        label="Trained Agent",
    )
    all_results["final_evaluation"] = eval_results

    # -----------------------------------------------------------------------
    # Compute improvements
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 60)

    baseline_reward = baseline_fixed["mean_reward"]
    trained_reward = eval_results["mean_reward"]
    reward_improvement = (trained_reward - baseline_reward) / max(abs(baseline_reward), 1e-6) * 100

    baseline_waiting = baseline_fixed["mean_total_waiting_time"]
    trained_waiting = eval_results["mean_total_waiting_time"]
    if baseline_waiting > 0:
        waiting_reduction = (baseline_waiting - trained_waiting) / baseline_waiting * 100
    else:
        waiting_reduction = 0.0

    logger.info(f"  Baseline (fixed) mean reward:    {baseline_reward:.3f}")
    logger.info(f"  Trained agent mean reward:       {trained_reward:.3f}")
    logger.info(f"  Reward improvement:              {reward_improvement:+.1f}%")
    logger.info(f"")
    logger.info(f"  Baseline total waiting time:     {baseline_waiting:.1f}")
    logger.info(f"  Trained agent total waiting:     {trained_waiting:.1f}")
    logger.info(f"  Waiting time reduction:          {waiting_reduction:+.1f}%")

    all_results["improvements"] = {
        "reward_improvement_pct": float(reward_improvement),
        "waiting_time_reduction_pct": float(waiting_reduction),
        "baseline_reward": float(baseline_reward),
        "trained_reward": float(trained_reward),
        "baseline_waiting": float(baseline_waiting),
        "trained_waiting": float(trained_waiting),
    }

    total_time = time.time() - total_start
    all_results["total_training_time_seconds"] = total_time
    all_results["total_training_time_minutes"] = total_time / 60.0

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    # Save models
    torch.save(intersection_model.state_dict(), output_dir / "intersection_model.pt")
    torch.save(district_model.state_dict(), output_dir / "district_model.pt")
    torch.save(comm_net.state_dict(), output_dir / "communication_net.pt")
    logger.info(f"Models saved to: {output_dir}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Environment: {args.grid_rows}x{args.grid_cols} synthetic traffic grid")
    logger.info(f"  Intersections: {env.num_intersections}")
    logger.info(f"  Districts: {env.num_districts}")
    logger.info(f"  Phase 1 (Pre-train): {pretrain_results.get('total_experience', pretrain_results.get('total_steps', 0))} experience, "
                f"best reward: {pretrain_results['best_mean_reward']:.3f}")
    logger.info(f"  Phase 2 (District):  {district_results['total_steps']} steps, "
                f"best reward: {district_results['best_mean_reward']:.3f}")
    logger.info(f"  Phase 3 (Joint):     {joint_results['total_steps']} steps, "
                f"best reward: {joint_results['best_mean_reward']:.3f}")
    logger.info(f"  Final Mean Reward:   {eval_results['mean_reward']:.3f}")
    logger.info(f"  Reward Improvement:  {reward_improvement:+.1f}% vs fixed baseline")
    logger.info(f"  Waiting Reduction:   {waiting_reduction:+.1f}%")
    logger.info(f"  Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Output Directory:    {output_dir}")
    logger.info("=" * 60)
    logger.info("Training completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
