"""Tests for AutomlEstimator."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from autopilot_explainer.estimator import AutomlEstimator


@pytest.fixture
def mock_sagemaker_session():
    """Create mock SageMaker session."""
    session = Mock()
    return session


@pytest.fixture
def mock_predictor():
    """Create mock predictor."""
    predictor = Mock()
    return predictor


@pytest.fixture
def estimator(mock_sagemaker_session, mock_predictor, monkeypatch):
    """Create AutomlEstimator with mocked dependencies."""

    def mock_predictor_init(*args, **kwargs):
        return mock_predictor

    monkeypatch.setattr(
        "autopilot_explainer.estimator.Predictor",
        mock_predictor_init
    )

    return AutomlEstimator(
        endpoint_name="test-endpoint",
        sagemaker_session=mock_sagemaker_session,
    )


class TestAutomlEstimator:
    """Test cases for AutomlEstimator."""

    def test_init(self, estimator):
        """Test estimator initialization."""
        assert estimator.endpoint_name == "test-endpoint"
        assert estimator.predictor is not None

    def test_get_automl_response_numpy(self, estimator, mock_predictor):
        """Test get_automl_response with numpy array."""
        mock_predictor.predict.return_value = b"0.5,0.6\n0.7,0.8\n"

        x = np.array([[1, 2], [3, 4]])
        response = estimator.get_automl_response(x)

        assert response == "0.5,0.6\n0.7,0.8\n"
        mock_predictor.predict.assert_called_once()

    def test_get_automl_response_dataframe(self, estimator, mock_predictor):
        """Test get_automl_response with DataFrame."""
        mock_predictor.predict.return_value = b"0.5,0.6\n"

        x = pd.DataFrame([[1, 2]], columns=["a", "b"])
        response = estimator.get_automl_response(x)

        assert response == "0.5,0.6\n"
        mock_predictor.predict.assert_called_once()

    def test_predict(self, estimator, mock_predictor):
        """Test predict method."""
        mock_predictor.predict.return_value = b"0,0.6\n1,0.8\n"

        x = np.array([[1, 2], [3, 4]])
        predictions = estimator.predict(x)

        assert len(predictions) == 2
        assert predictions[0] == "0"
        assert predictions[1] == "1"

    def test_predict_proba(self, estimator, mock_predictor):
        """Test predict_proba method."""
        mock_predictor.predict.return_value = b"0,0.6\n1,0.8\n"

        x = np.array([[1, 2], [3, 4]])
        probabilities = estimator.predict_proba(x)

        assert len(probabilities) == 2
        assert probabilities[0] == 0.6
        assert probabilities[1] == 0.8
        assert probabilities.dtype == float

    def test_get_automl_response_invalid_type(self, estimator):
        """Test get_automl_response with invalid input type."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            estimator.get_automl_response("invalid")
