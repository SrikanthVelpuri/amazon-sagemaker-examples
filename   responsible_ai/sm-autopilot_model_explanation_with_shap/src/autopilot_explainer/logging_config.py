"""Logging configuration utilities."""

import logging
import sys
from .config import LoggingConfig


def setup_logging(config: LoggingConfig = None) -> None:
    """Setup logging configuration.

    Args:
        config: LoggingConfig instance (default: load from environment)
    """
    if config is None:
        config = LoggingConfig.from_env()

    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        datefmt=config.date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set library loggers to WARNING to reduce noise
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
