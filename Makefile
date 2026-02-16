.PHONY: help install install-dev install-mamba quick-check test clean clean-all

install:
	@echo "Installing core dependencies..."
	python3 -m pip install -e "."

install-dev:
	@echo "Installing with dev tools..."
	python3 -m pip install -e ".[dev]"

install-mamba:
	@echo "Installing Mamba support (this may take several minutes)..."
	python3 -m pip install -e ".[mamba]" || echo "Mamba installation failed (optional)"

quick-check:
	@echo "Running quick validation checks..."
	python3 quick_check.py

test: 
	@echo "Running full test suite..."
	python3 test_e2e.py

train:
	@echo "Training with custom config..."
	@echo "Usage: make train CONFIG=hsi_compression/configs/models/tcn_lossless.yaml"
	python3 train.py --config $(CONFIG)

clean:
	@echo "Cleaning up..."
	rm -rf checkpoints/ results/ __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cleanup complete"

clean-all: clean
	@echo "Removing all generated data..."
	rm -rf dummy_hyspecnet11k/ test_output.log .wandb/
	@echo "Complete cleanup done"