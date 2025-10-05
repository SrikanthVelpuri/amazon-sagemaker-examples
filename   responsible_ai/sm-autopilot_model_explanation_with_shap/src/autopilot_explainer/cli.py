"""Command-line interface for AutoPilot Explainer."""

import argparse
import sys
import json
import pandas as pd
import sagemaker
from pathlib import Path

from .estimator import AutomlEstimator
from .explainer import ShapExplainer
from .endpoint_manager import ManagedEndpoint
from .config import ExplainerConfig, EndpointConfig
from .logging_config import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for SageMaker Autopilot models"
    )

    parser.add_argument(
        "--endpoint-name", required=True, help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--data-file", required=True, help="Path to CSV data file"
    )
    parser.add_argument(
        "--problem-type",
        required=True,
        choices=["Regression", "BinaryClassification", "MulticlassClassification"],
        help="Problem type",
    )
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "global"],
        help="Explanation mode: local or global (default: local)",
    )
    parser.add_argument(
        "--output", required=True, help="Output file path for SHAP values (JSON)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples to explain (default: 1)",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=50,
        help="Number of background samples (default: 50)",
    )
    parser.add_argument(
        "--auto-delete",
        action="store_true",
        help="Auto-delete endpoint after completion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--region", help="AWS region (default: from boto3 session)"
    )

    args = parser.parse_args()

    # Setup logging
    from .config import LoggingConfig
    setup_logging(LoggingConfig(level=args.log_level))

    try:
        # Load data
        print(f"Loading data from {args.data_file}...")
        data = pd.read_csv(args.data_file)

        # Initialize SageMaker session
        session = sagemaker.Session()

        # Create estimator
        print(f"Connecting to endpoint: {args.endpoint_name}")
        estimator = AutomlEstimator(
            endpoint_name=args.endpoint_name,
            sagemaker_session=session,
        )

        # Create explainer
        print("Initializing SHAP explainer...")
        explainer = ShapExplainer(
            estimator=estimator,
            background_data=data,
            problem_type=args.problem_type,
            n_background_samples=args.n_background,
        )

        # Generate explanations
        with ManagedEndpoint(
            args.endpoint_name,
            auto_delete=args.auto_delete,
            region_name=args.region,
        ):
            if args.mode == "local":
                print(f"Generating local explanations for {args.n_samples} sample(s)...")
                X = data.iloc[:args.n_samples]
                shap_values = explainer.explain_local(X)
            else:
                print(f"Generating global explanation with {args.n_samples} samples...")
                shap_values, X = explainer.explain_global(
                    data, n_samples=args.n_samples
                )

        # Save results
        print(f"Saving results to {args.output}...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "mode": args.mode,
            "problem_type": args.problem_type,
            "expected_value": explainer.get_expected_value(),
            "shap_values": shap_values.tolist(),
            "feature_names": list(X.columns) if isinstance(X, pd.DataFrame) else None,
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        # Calculate and display feature importance
        if isinstance(X, pd.DataFrame):
            importance = explainer.get_feature_importance(
                shap_values, feature_names=list(X.columns)
            )
            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        print(f"\n✓ Successfully saved SHAP explanations to {args.output}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
