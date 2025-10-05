"""SHAP-based explainer for AutoML models."""

import logging
from typing import Optional, Union, Literal
import pandas as pd
import numpy as np
from shap import KernelExplainer, sample
from scipy.special import expit

from .estimator import AutomlEstimator

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP explainer for SageMaker Autopilot models.

    Provides local and global model explanations using SHAP KernelExplainer.

    Args:
        estimator: AutomlEstimator instance
        background_data: Background data for SHAP (DataFrame or ndarray)
        problem_type: "Regression" or "BinaryClassification"/"MulticlassClassification"
        n_background_samples: Number of samples to use for background (default: 50)
    """

    def __init__(
        self,
        estimator: AutomlEstimator,
        background_data: Union[pd.DataFrame, np.ndarray],
        problem_type: str,
        n_background_samples: int = 50,
    ):
        self.estimator = estimator
        self.problem_type = problem_type

        # Sample background data if needed
        if len(background_data) > n_background_samples:
            logger.info(f"Sampling {n_background_samples} background samples")
            self.background_data = sample(background_data, n_background_samples)
        else:
            self.background_data = background_data

        # Determine link function based on problem type
        self.link = "identity" if problem_type == "Regression" else "logit"

        # Choose prediction function
        if problem_type == "Regression":
            predict_fn = estimator.predict
        else:
            predict_fn = estimator.predict_proba

        # Initialize KernelExplainer
        logger.info(f"Initializing KernelExplainer with link={self.link}")
        self.explainer = KernelExplainer(predict_fn, self.background_data, link=self.link)

        logger.info(
            f"Expected value: {self.get_expected_value()} "
            f"(raw: {self.explainer.expected_value})"
        )

    def get_expected_value(self) -> float:
        """Get expected value (baseline prediction).

        For classification, converts from log-odds to probability.

        Returns:
            Expected prediction value
        """
        if self.link == "logit":
            return float(expit(self.explainer.expected_value))
        return float(self.explainer.expected_value)

    def explain_local(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        nsamples: Union[int, str] = "auto",
        l1_reg: Union[str, float] = "aic",
    ) -> np.ndarray:
        """Generate local SHAP explanations for samples.

        Args:
            x: Input samples to explain
            nsamples: Number of samples for KernelSHAP (default: "auto")
            l1_reg: Regularization method ("aic", "bic", or float) or "num_features(k)"

        Returns:
            SHAP values array
        """
        try:
            logger.info(f"Calculating SHAP values for {len(x)} sample(s)")
            shap_values = self.explainer.shap_values(x, nsamples=nsamples, l1_reg=l1_reg)
            logger.info("SHAP values calculated successfully")
            return shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            raise

    def explain_global(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        n_samples: int = 50,
        nsamples: Union[int, str] = "auto",
        l1_reg: Union[str, float] = "aic",
    ) -> tuple[np.ndarray, Union[pd.DataFrame, np.ndarray]]:
        """Generate global SHAP explanations by aggregating local explanations.

        Args:
            data: Full dataset to sample from
            n_samples: Number of samples to explain (default: 50)
            nsamples: Number of samples for KernelSHAP (default: "auto")
            l1_reg: Regularization method

        Returns:
            Tuple of (shap_values, sampled_data)
        """
        try:
            # Sample data
            logger.info(f"Sampling {n_samples} samples for global explanation")
            X = sample(data, n_samples)

            # Calculate SHAP values
            shap_values = self.explain_local(X, nsamples=nsamples, l1_reg=l1_reg)

            return shap_values, X
        except Exception as e:
            logger.error(f"Error calculating global explanation: {str(e)}")
            raise

    def get_feature_importance(
        self, shap_values: np.ndarray, feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """Calculate feature importance from SHAP values.

        Args:
            shap_values: SHAP values from explain_local or explain_global
            feature_names: Optional list of feature names

        Returns:
            DataFrame with feature importance scores
        """
        try:
            # Calculate mean absolute SHAP value for each feature
            importance = np.abs(shap_values).mean(axis=0)

            # Create DataFrame
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]

            df = pd.DataFrame(
                {"feature": feature_names, "importance": importance}
            ).sort_values("importance", ascending=False)

            logger.info(f"Calculated importance for {len(feature_names)} features")
            return df
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
