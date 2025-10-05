"""Configuration management for AutoPilot Explainer."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplainerConfig:
    """Configuration for SHAP explainer.

    Attributes:
        n_background_samples: Number of background samples for SHAP
        nsamples: Number of samples for KernelSHAP ("auto" or integer)
        l1_reg: L1 regularization for SHAP ("aic", "bic", float, or "num_features(k)")
        n_global_samples: Number of samples for global explanation
    """

    n_background_samples: int = 50
    nsamples: str = "auto"
    l1_reg: str = "aic"
    n_global_samples: int = 50

    @classmethod
    def from_env(cls) -> "ExplainerConfig":
        """Load configuration from environment variables."""
        return cls(
            n_background_samples=int(
                os.getenv("SHAP_BACKGROUND_SAMPLES", cls.n_background_samples)
            ),
            nsamples=os.getenv("SHAP_NSAMPLES", cls.nsamples),
            l1_reg=os.getenv("SHAP_L1_REG", cls.l1_reg),
            n_global_samples=int(os.getenv("SHAP_GLOBAL_SAMPLES", cls.n_global_samples)),
        )


@dataclass
class EndpointConfig:
    """Configuration for SageMaker endpoint.

    Attributes:
        endpoint_name: Name of SageMaker endpoint
        region_name: AWS region
        auto_delete: Whether to auto-delete endpoint after use
        content_type: Content type for requests
        accept: Accept type for responses
    """

    endpoint_name: str
    region_name: Optional[str] = None
    auto_delete: bool = False
    content_type: str = "text/csv"
    accept: str = "text/csv"

    @classmethod
    def from_env(cls, endpoint_name: Optional[str] = None) -> "EndpointConfig":
        """Load configuration from environment variables."""
        return cls(
            endpoint_name=endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME"),
            region_name=os.getenv("AWS_REGION"),
            auto_delete=os.getenv("AUTO_DELETE_ENDPOINT", "false").lower() == "true",
            content_type=os.getenv("ENDPOINT_CONTENT_TYPE", cls.content_type),
            accept=os.getenv("ENDPOINT_ACCEPT", cls.accept),
        )


@dataclass
class LoggingConfig:
    """Configuration for logging.

    Attributes:
        level: Logging level
        format: Log format string
        date_format: Date format string
    """

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Load configuration from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", cls.level),
            format=os.getenv("LOG_FORMAT", cls.format),
            date_format=os.getenv("LOG_DATE_FORMAT", cls.date_format),
        )
