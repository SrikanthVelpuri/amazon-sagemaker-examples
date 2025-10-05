"""Tests for ShapExplainer."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from autopilot_explainer.explainer import ShapExplainer


@pytest.fixture
def mock_estimator():
    """Create mock AutomlEstimator."""
    estimator = Mock()
    estimator.predict.return_value = np.array([0.5, 0.6, 0.7])
    estimator.predict_proba.return_value = np.array([0.5, 0.6, 0.7])
    return estimator


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "feature3": [100, 200, 300, 400, 500],
    })


class TestShapExplainer:
    """Test cases for ShapExplainer."""

    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_init_regression(self, mock_kernel_explainer, mock_estimator, sample_data):
        """Test explainer initialization for regression."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_kernel_explainer.return_value = mock_explainer_instance

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="Regression",
            n_background_samples=3,
        )

        assert explainer.problem_type == "Regression"
        assert explainer.link == "identity"
        mock_kernel_explainer.assert_called_once()
        # Verify predict (not predict_proba) was passed
        call_args = mock_kernel_explainer.call_args
        assert call_args[0][0] == mock_estimator.predict

    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_init_classification(self, mock_kernel_explainer, mock_estimator, sample_data):
        """Test explainer initialization for classification."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_kernel_explainer.return_value = mock_explainer_instance

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="BinaryClassification",
            n_background_samples=3,
        )

        assert explainer.problem_type == "BinaryClassification"
        assert explainer.link == "logit"
        # Verify predict_proba was passed
        call_args = mock_kernel_explainer.call_args
        assert call_args[0][0] == mock_estimator.predict_proba

    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_get_expected_value_regression(
        self, mock_kernel_explainer, mock_estimator, sample_data
    ):
        """Test get_expected_value for regression."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_kernel_explainer.return_value = mock_explainer_instance

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="Regression",
        )

        assert explainer.get_expected_value() == 0.5

    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_get_expected_value_classification(
        self, mock_kernel_explainer, mock_estimator, sample_data
    ):
        """Test get_expected_value for classification (logit conversion)."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.0  # logit of 0.5
        mock_kernel_explainer.return_value = mock_explainer_instance

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="BinaryClassification",
        )

        # Should be close to 0.5 after expit transformation
        assert abs(explainer.get_expected_value() - 0.5) < 0.01

    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_explain_local(
        self, mock_kernel_explainer, mock_estimator, sample_data
    ):
        """Test local explanation."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_explainer_instance.shap_values.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_kernel_explainer.return_value = mock_explainer_instance

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="Regression",
        )

        x = sample_data.iloc[0:1]
        shap_values = explainer.explain_local(x)

        assert shap_values.shape == (1, 3)
        mock_explainer_instance.shap_values.assert_called_once()

    @patch("autopilot_explainer.explainer.sample")
    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_explain_global(
        self, mock_kernel_explainer, mock_sample, mock_estimator, sample_data
    ):
        """Test global explanation."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_explainer_instance.shap_values.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        mock_kernel_explainer.return_value = mock_explainer_instance

        # Mock sample to return first 2 rows
        mock_sample.return_value = sample_data.iloc[:2]

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="Regression",
        )

        shap_values, X = explainer.explain_global(sample_data, n_samples=2)

        assert shap_values.shape == (2, 3)
        assert len(X) == 2

    @patch("autopilot_explainer.explainer.KernelExplainer")
    def test_get_feature_importance(
        self, mock_kernel_explainer, mock_estimator, sample_data
    ):
        """Test feature importance calculation."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_kernel_explainer.return_value = mock_explainer_instance

        explainer = ShapExplainer(
            estimator=mock_estimator,
            background_data=sample_data,
            problem_type="Regression",
        )

        shap_values = np.array([
            [0.1, 0.5, 0.2],
            [0.2, 0.4, 0.1],
        ])

        importance = explainer.get_feature_importance(
            shap_values, feature_names=["f1", "f2", "f3"]
        )

        assert len(importance) == 3
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        # f2 should be most important (mean abs = 0.45)
        assert importance.iloc[0]["feature"] == "f2"
