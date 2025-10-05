"""AutoML Estimator for SageMaker Autopilot models."""

import logging
from typing import Union
import numpy as np
import pandas as pd
from sagemaker.predictor import Predictor
import sagemaker

logger = logging.getLogger(__name__)


class AutomlEstimator:
    """Wrapper for SageMaker Autopilot inference endpoint.

    Provides unified predict and predict_proba interfaces for both
    regression and classification tasks.

    Args:
        endpoint_name: Name of the deployed SageMaker endpoint
        sagemaker_session: Active SageMaker session
        content_type: Content type for requests (default: "text/csv")
        accept: Accept type for responses (default: "text/csv")
    """

    def __init__(
        self,
        endpoint_name: str,
        sagemaker_session,
        content_type: str = "text/csv",
        accept: str = "text/csv",
    ):
        self.endpoint_name = endpoint_name
        self.predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=sagemaker.serializers.CSVSerializer(),
            content_type=content_type,
            accept=accept,
        )
        logger.info(f"Initialized AutomlEstimator for endpoint: {endpoint_name}")

    def get_automl_response(self, x: Union[np.ndarray, pd.DataFrame]) -> str:
        """Get raw response from AutoML endpoint.

        Args:
            x: Input data as numpy array or pandas DataFrame

        Returns:
            Raw CSV response string from endpoint
        """
        try:
            if isinstance(x, np.ndarray):
                payload = ""
                for row in x:
                    payload = payload + ",".join(map(str, row)) + "\n"
            elif isinstance(x, pd.DataFrame):
                payload = x.to_csv(sep=",", header=False, index=False)
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")

            response = self.predictor.predict(payload).decode("utf-8")
            logger.debug(f"Received response from endpoint: {len(response)} bytes")
            return response
        except Exception as e:
            logger.error(f"Error getting AutoML response: {str(e)}")
            raise

    def predict(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get predictions for regression tasks.

        Args:
            x: Input data as numpy array or pandas DataFrame

        Returns:
            Array of prediction values (or labels for classification)
        """
        try:
            response = self.get_automl_response(x)
            # First column contains numeric prediction value or label
            predictions = np.array([row.split(",")[0] for row in response.split("\n")[:-1]])
            logger.debug(f"Generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get class probabilities for classification tasks.

        Args:
            x: Input data as numpy array or pandas DataFrame

        Returns:
            Array of class probabilities
        """
        try:
            response = self.get_automl_response(x)
            # Second column contains class probability
            probabilities = np.array([row.split(",")[1] for row in response.split("\n")[:-1]])
            logger.debug(f"Generated {len(probabilities)} probability predictions")
            return probabilities.astype(float)
        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            raise
