.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=autopilot_explainer --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ examples/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make install  - Install package with dev dependencies"
	@echo "  make test     - Run tests with coverage"
	@echo "  make lint     - Run linters (flake8, mypy)"
	@echo "  make format   - Format code with black"
	@echo "  make clean    - Remove build artifacts"
