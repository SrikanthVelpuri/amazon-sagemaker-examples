"""Example usage of the autopilot_explainer package."""

import pandas as pd
import sagemaker
from autopilot_explainer import AutomlEstimator, ShapExplainer, ManagedEndpoint
from autopilot_explainer.logging_config import setup_logging

# Setup logging
setup_logging()

# Configuration
ENDPOINT_NAME = "your-autopilot-endpoint"
DATA_FILE = "../churn.txt"
PROBLEM_TYPE = "BinaryClassification"  # or "Regression", "MulticlassClassification"

# Load data
print("Loading data...")
churn_data = pd.read_csv(DATA_FILE)
data_without_target = churn_data.drop(columns=["Churn?"])

# Initialize SageMaker session
session = sagemaker.Session()

# Create estimator
print(f"Connecting to endpoint: {ENDPOINT_NAME}")
estimator = AutomlEstimator(
    endpoint_name=ENDPOINT_NAME,
    sagemaker_session=session,
)

# Create explainer
print("Initializing SHAP explainer...")
explainer = ShapExplainer(
    estimator=estimator,
    background_data=data_without_target,
    problem_type=PROBLEM_TYPE,
    n_background_samples=50,
)

print(f"Expected baseline prediction: {explainer.get_expected_value():.4f}")

# Example 1: Local explanation for a single sample
print("\n=== Local Explanation ===")
with ManagedEndpoint(ENDPOINT_NAME):
    x_single = data_without_target.iloc[0:1]
    shap_values_local = explainer.explain_local(x_single, l1_reg="num_features(5)")

    # Get feature importance
    importance_local = explainer.get_feature_importance(
        shap_values_local, feature_names=list(x_single.columns)
    )
    print("\nTop 5 influential features for this sample:")
    print(importance_local.head(5))

# Example 2: Global explanation
print("\n=== Global Explanation ===")
with ManagedEndpoint(ENDPOINT_NAME):
    shap_values_global, X_global = explainer.explain_global(
        data_without_target, n_samples=50
    )

    # Get feature importance
    importance_global = explainer.get_feature_importance(
        shap_values_global, feature_names=list(X_global.columns)
    )
    print("\nTop 10 globally important features:")
    print(importance_global.head(10))

# Example 3: Using auto-delete
print("\n=== Using Auto-Delete ===")
with ManagedEndpoint(ENDPOINT_NAME, auto_delete=False):  # Set to True to delete
    x_sample = data_without_target.iloc[:3]
    shap_values = explainer.explain_local(x_sample)
    print(f"Generated SHAP values for {len(x_sample)} samples")

print("\n✓ Done!")
