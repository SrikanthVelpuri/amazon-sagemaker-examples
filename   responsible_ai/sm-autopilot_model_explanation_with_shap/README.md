# AutoPilot Explainer

A production-ready Python package for generating SHAP-based explanations for AWS SageMaker Autopilot models.

## Features

- **SHAP Integration**: Uses KernelExplainer for model-agnostic explanations
- **Local & Global Explanations**: Support for both individual predictions and model-wide insights
- **Production Ready**: Comprehensive error handling, logging, and configuration management
- **CLI & API**: Use via command-line interface or programmatically
- **Tested**: Full unit test coverage with pytest
- **Type Safe**: Type hints throughout the codebase

## Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
import pandas as pd
import sagemaker
from autopilot_explainer import AutomlEstimator, ShapExplainer, ManagedEndpoint

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize SageMaker session and estimator
session = sagemaker.Session()
estimator = AutomlEstimator(
    endpoint_name="your-endpoint-name",
    sagemaker_session=session,
)

# Create explainer
explainer = ShapExplainer(
    estimator=estimator,
    background_data=data,
    problem_type="BinaryClassification",  # or "Regression"
    n_background_samples=50,
)

# Generate local explanation for a single sample
with ManagedEndpoint("your-endpoint-name"):
    x_sample = data.iloc[0:1]
    shap_values = explainer.explain_local(x_sample)

    # Get feature importance
    importance = explainer.get_feature_importance(
        shap_values,
        feature_names=list(data.columns)
    )
    print(importance.head(10))
```

### Command Line Interface

```bash
# Generate local explanation
autopilot-explain \
    --endpoint-name my-endpoint \
    --data-file data.csv \
    --problem-type BinaryClassification \
    --mode local \
    --output results/shap_values.json \
    --n-samples 1

# Generate global explanation
autopilot-explain \
    --endpoint-name my-endpoint \
    --data-file data.csv \
    --problem-type Regression \
    --mode global \
    --output results/global_shap.json \
    --n-samples 50 \
    --n-background 100
```

## Configuration

### Environment Variables

```bash
# SageMaker endpoint configuration
export SAGEMAKER_ENDPOINT_NAME=my-endpoint
export AWS_REGION=us-west-2
export AUTO_DELETE_ENDPOINT=false

# SHAP configuration
export SHAP_BACKGROUND_SAMPLES=50
export SHAP_NSAMPLES=auto
export SHAP_L1_REG=aic
export SHAP_GLOBAL_SAMPLES=50

# Logging configuration
export LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from autopilot_explainer.config import ExplainerConfig, EndpointConfig

# Configure explainer
explainer_config = ExplainerConfig(
    n_background_samples=100,
    nsamples="auto",
    l1_reg="num_features(10)",  # Only top 10 features
    n_global_samples=75,
)

# Configure endpoint
endpoint_config = EndpointConfig(
    endpoint_name="my-endpoint",
    region_name="us-west-2",
    auto_delete=False,
)
```

## Package Structure

```
autopilot-explainer/
├── src/autopilot_explainer/
│   ├── __init__.py           # Package exports
│   ├── estimator.py          # AutoML estimator wrapper
│   ├── explainer.py          # SHAP explainer
│   ├── endpoint_manager.py   # Endpoint context manager
│   ├── config.py             # Configuration classes
│   ├── logging_config.py     # Logging setup
│   └── cli.py                # Command-line interface
├── tests/                    # Unit tests
├── examples/                 # Example scripts
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=autopilot_explainer --cov-report=html

# Run specific test file
pytest tests/test_explainer.py
```

## Advanced Usage

### Custom Background Data

```python
from shap import sample

# Sample background data strategically
background = sample(full_data, 100)

explainer = ShapExplainer(
    estimator=estimator,
    background_data=background,
    problem_type="BinaryClassification",
    n_background_samples=100,
)
```

### Feature Selection with L1 Regularization

```python
# Only explain top 5 features
shap_values = explainer.explain_local(
    x_sample,
    l1_reg="num_features(5)"
)
```

### Auto-Delete Endpoints

```python
# Automatically delete endpoint after use
with ManagedEndpoint("temp-endpoint", auto_delete=True):
    shap_values = explainer.explain_local(x_sample)
# Endpoint is deleted here
```

## Best Practices

1. **Background Data Selection**: Use a representative sample of your data (50-100 samples typically)
2. **Computation Cost**: Each SHAP calculation requires many inference calls. For a single sample with 50 background samples and nsamples="auto", expect ~100K+ inference calls
3. **Feature Selection**: Use `l1_reg="num_features(k)"` to focus on top-k features and reduce computation
4. **Endpoint Management**: Use `ManagedEndpoint` context manager to ensure proper endpoint lifecycle
5. **Logging**: Enable DEBUG logging for troubleshooting: `export LOG_LEVEL=DEBUG`

## Problem Types

- **Regression**: Use `problem_type="Regression"` and `estimator.predict()`
- **Binary Classification**: Use `problem_type="BinaryClassification"` and `estimator.predict_proba()`
- **Multiclass Classification**: Use `problem_type="MulticlassClassification"` and `estimator.predict_proba()`

## SHAP Values Interpretation

- **Positive SHAP value**: Feature pushes prediction higher
- **Negative SHAP value**: Feature pushes prediction lower
- **Magnitude**: Indicates feature's impact strength
- **Expected value**: Baseline prediction with all features missing

For classification, SHAP values are in log-odds space. Use `scipy.special.expit()` to convert to probabilities.

## Troubleshooting

### Endpoint Not In Service

```
RuntimeError: Endpoint my-endpoint is not InService (status: Creating)
```
Wait for endpoint to reach InService status before running explainer.

### Out of Memory

Reduce background samples or use feature selection:
```python
explainer = ShapExplainer(..., n_background_samples=25)
shap_values = explainer.explain_local(x, l1_reg="num_features(5)")
```

### Slow Performance

KernelSHAP is computationally expensive. Consider:
- Reducing background samples
- Using feature selection with `l1_reg`
- Explaining fewer samples for global explanations

## License

Apache 2.0

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## References

- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)
- [AWS SageMaker Autopilot](https://aws.amazon.com/sagemaker/autopilot/)
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
