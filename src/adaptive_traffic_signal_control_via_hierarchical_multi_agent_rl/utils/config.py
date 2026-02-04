"""Configuration management utilities."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the traffic control system.

    This class provides a centralized way to manage configuration parameters
    for the hierarchical multi-agent reinforcement learning system.
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize configuration.

        Args:
            config_dict: Dictionary containing configuration parameters.
        """
        self._config = OmegaConf.create(config_dict)
        self._validate_config()

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]

    def __getattr__(self, key: str) -> Any:
        """Get configuration value as attribute."""
        if key.startswith("_"):
            return super().__getattribute__(key)
        return getattr(self._config, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default.

        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        try:
            return OmegaConf.select(self._config, key)
        except Exception:
            return default

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply.
        """
        self._config = OmegaConf.merge(self._config, updates)
        self._validate_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return OmegaConf.to_container(self._config, resolve=True)

    def save(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save configuration file.
        """
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = [
            "experiment",
            "environment",
            "agents",
            "training",
            "model",
        ]

        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate specific parameters
        if self._config.training.total_timesteps <= 0:
            raise ValueError("training.total_timesteps must be positive")

        if self._config.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")

        if not 0 < self._config.training.learning_rate < 1:
            raise ValueError("training.learning_rate must be between 0 and 1")

        if not 0 <= self._config.training.gamma <= 1:
            raise ValueError("training.gamma must be between 0 and 1")

        # Validate environment parameters
        if self._config.environment.simulation_time <= 0:
            raise ValueError("environment.simulation_time must be positive")

        if self._config.environment.step_length <= 0:
            raise ValueError("environment.step_length must be positive")

        # Validate agent parameters
        if self._config.agents.hierarchy_levels < 1:
            raise ValueError("agents.hierarchy_levels must be at least 1")

        logger.info("Configuration validation passed")


def load_config(
    config_path: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default config.
        config_overrides: Optional dictionary of configuration overrides.

    Returns:
        Loaded configuration object.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        yaml.YAMLError: If configuration file is invalid YAML.
    """
    if config_path is None:
        # Use default config
        package_dir = Path(__file__).parent.parent.parent.parent.parent
        config_path = package_dir / "configs" / "default.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")

    if config_overrides:
        config_dict.update(config_overrides)

    logger.info(f"Loaded configuration from {config_path}")
    return Config(config_dict)


def setup_logging(config: Config) -> None:
    """Set up logging based on configuration.

    Args:
        config: Configuration object.
    """
    import colorlog

    log_level = getattr(logging, config.experiment.log_level.upper())

    # Create formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Set up handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        force=True
    )

    # Set specific logger levels
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger.info(f"Logging configured with level: {config.experiment.log_level}")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seeds set to {seed}")


def create_output_directory(config: Config) -> str:
    """Create output directory for experiment.

    Args:
        config: Configuration object.

    Returns:
        Path to created output directory.
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.experiment.output_dir}/{config.experiment.name}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Save config to output directory
    config.save(f"{output_dir}/config.yaml")

    logger.info(f"Created output directory: {output_dir}")
    return output_dir