"""Tests for configuration management."""

import pytest
import os
from autopilot_explainer.config import (
    ExplainerConfig,
    EndpointConfig,
    LoggingConfig,
)


class TestExplainerConfig:
    """Test cases for ExplainerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExplainerConfig()
        assert config.n_background_samples == 50
        assert config.nsamples == "auto"
        assert config.l1_reg == "aic"
        assert config.n_global_samples == 50

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("SHAP_BACKGROUND_SAMPLES", "100")
        monkeypatch.setenv("SHAP_NSAMPLES", "1000")
        monkeypatch.setenv("SHAP_L1_REG", "bic")
        monkeypatch.setenv("SHAP_GLOBAL_SAMPLES", "75")

        config = ExplainerConfig.from_env()
        assert config.n_background_samples == 100
        assert config.nsamples == "1000"
        assert config.l1_reg == "bic"
        assert config.n_global_samples == 75


class TestEndpointConfig:
    """Test cases for EndpointConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EndpointConfig(endpoint_name="test-endpoint")
        assert config.endpoint_name == "test-endpoint"
        assert config.region_name is None
        assert config.auto_delete is False
        assert config.content_type == "text/csv"
        assert config.accept == "text/csv"

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "env-endpoint")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setenv("AUTO_DELETE_ENDPOINT", "true")

        config = EndpointConfig.from_env()
        assert config.endpoint_name == "env-endpoint"
        assert config.region_name == "us-west-2"
        assert config.auto_delete is True


class TestLoggingConfig:
    """Test cases for LoggingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "%(asctime)s" in config.format
        assert config.date_format == "%Y-%m-%d %H:%M:%S"

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = LoggingConfig.from_env()
        assert config.level == "DEBUG"
