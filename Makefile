.PHONY: help install install-dev test test-e2e test-quick train eval clean data docs lint format check-env

check-env:
	@echo "Checking environment..."
	@python3 --version
	@python3 -c "import torch; print('PyTorch', torch.__version__)" || echo "PyTorch not found"
	@python3 -c "import omegaconf; print('OmegaConf')" || echo "OmegaConf not found"
	@python3 -c "import torchvision; print('TorchVision')" || echo "TorchVision not found"
	@python3 -c "import scipy; print('SciPy')" || echo "SciPy not found"
	@echo ""
	@echo "Directory structure:"
	@ls -d hsi_compression/* 2>/dev/null | sed 's/^/  /'
	@echo ""

install:
	@echo "Installing core dependencies from pyproject.toml..."
	python3 -m pip install -e "."
	@echo "Core dependencies installed"

install-dev:
	@echo "Installing with development tools from pyproject.toml..."
	python3 -m pip install -e ".[dev]"
	@echo "Development dependencies installed"

install-experiments:
	@echo "Installing with experiment tracking (WandB, TensorBoard)..."
	python3 -m pip install -e ".[experiments]"
	@echo "Experiment tracking dependencies installed"

install-mamba:
	@echo "Installing with Mamba model support (may take several minutes)..."
	python3 -m pip install -e ".[mamba]" || echo "Mamba installation failed (optional dependency)"
	@echo "Mamba dependencies installed (if successful)"

install-all:
	@echo "Installing all optional dependencies (dev + experiments + mamba)..."
	python3 -m pip install -e ".[all]" || echo "Some optional dependencies failed"
	@echo "All dependencies installed (if successful)"

data:
	@echo "Generating dummy HySpecNet-11k dataset..."
	python3 generate_dummy_data.py \
		--output ./dummy_hyspecnet11k \
		--num_scenes 3 \
		--patches_per_scene 5 \
		--num_bands 224 \
		--patch_size 32
	@echo "Dataset generated at ./dummy_hyspecnet11k"
	@ls -la dummy_hyspecnet11k/patches/ | tail -n +4 | wc -l | xargs echo "  Total patches:"

data-clean:
	@echo "Removing dummy dataset..."
	rm -rf dummy_hyspecnet11k
	@echo "Dataset removed"

test: data
	@echo "Running E2E test suite..."
	python3 test_e2e.py

test-quick:
	@echo "Running quick sanity check..."
	@echo "1. Testing imports..."
	@python3 -c "from hsi_compression.models import get_model; from hsi_compression.datasets import get_dataset; print('Imports work')"
	@echo "2. Testing model loading..."
	@python3 -c "from hsi_compression.models import list_models; models = list_models(); print(f'Found {len(models)} models: {models}')"
	@echo "3. Testing config loading..."
	@python3 -c "from hsi_compression.utils.config import load_config; config = load_config('hsi_compression/configs/models/tcn_lossless.yaml'); print(f'Config loaded with seed={config.seed}')"
	@echo ""
	@echo "Quick sanity check passed!"

test-dataset:
	@echo "Testing dataset loading..."
	python3 -c "\
		from hsi_compression.datasets import get_dataset, get_default_transforms; \
		from torch.utils.data import DataLoader; \
		print('Loading dataset...'); \
		dataset = get_dataset('hyspecnet11k', root_dir='./dummy_hyspecnet11k', split='train', transform=get_default_transforms('lossy')); \
		print(f'Dataset loaded: {len(dataset)} patches'); \
		sample = dataset[0]; \
		print(f'Sample shape: {sample.shape}'); \
		loader = DataLoader(dataset, batch_size=2); \
		batch = next(iter(loader)); \
		print(f'Batch shape: {batch.shape}'); \
		print('Dataset test passed!')"

test-models:
	@echo "Testing model initialization..."
	python3 -c "\
		import torch; \
		from hsi_compression.models import get_model; \
		print('Testing TCN...'); \
		tcn = get_model('tcn_lossless', num_bands=224); \
		print(f'TCN loaded: {sum(p.numel() for p in tcn.parameters()):,} parameters'); \
		x = torch.randn(2, 224, 32, 32); \
		out = tcn(x); \
		print(f'TCN forward pass works'); \
		print('Model tests passed!')"

test-metrics:
	@echo "Testing metrics..."
	python3 -c "\
		import torch; \
		from hsi_compression.metrics import get_metric; \
		x = torch.randn(2, 224, 32, 32); \
		y = x + torch.randn_like(x) * 0.01; \
		psnr = get_metric('psnr'); \
		ssim = get_metric('ssim'); \
		sam = get_metric('sam'); \
		print(f'PSNR: {psnr(x, y):.2f} dB'); \
		print(f'SSIM: {ssim(x, y):.4f}'); \
		print(f'SAM: {sam(x, y):.4f}°'); \
		print('Metrics tests passed!')"

test-losses:
	@echo "Testing loss functions..."
	python3 -c "\
		import torch; \
		from hsi_compression.losses import get_distortion_loss; \
		x = torch.randn(2, 224, 32, 32, requires_grad=True); \
		y = x + torch.randn_like(x) * 0.01; \
		mse_loss = get_distortion_loss('mse'); \
		sam_loss = get_distortion_loss('sam'); \
		loss_mse = mse_loss(x, y); \
		loss_sam = sam_loss(x, y); \
		print(f'MSE Loss: {loss_mse.item():.6f}'); \
		print(f'SAM Loss: {loss_sam.item():.6f}'); \
		loss_mse.backward(); \
		print('Backward pass works'); \
		print('Loss tests passed!')"

train-tcn: data
	@echo "Training TCN lossless model (5 epochs on dummy data)..."
	python3 train.py \
		--config hsi_compression/configs/models/tcn_lossless.yaml \
		--overrides \
			data.root_dir=./dummy_hyspecnet11k \
			training.epochs=5 \
			data.batch_size=2 \
			training.optimizer.lr=0.001
	@echo "Training completed. Checkpoint saved to ./checkpoints/"

train-tcn-full:
	@echo "Training TCN lossless model (100 epochs on real data)..."
	@echo "Make sure to set data.root_dir to your HySpecNet-11k path"
	python3 train.py \
		--config hsi_compression/configs/models/tcn_lossless.yaml \
		--overrides \
			data.root_dir=/path/to/hyspecnet11k \
			training.epochs=100 \
			data.batch_size=8

train-mamba: data
	@echo "Training Mamba lossy model (5 epochs on dummy data)..."
	@echo "Requires mamba-ssm. Run 'make install-mamba' if not installed."
	python3 train.py \
		--config hsi_compression/configs/models/mamba_lossy.yaml \
		--overrides \
			data.root_dir=./dummy_hyspecnet11k \
			training.epochs=5 \
			data.batch_size=2 \
			training.lambda=0.01

train-custom:
	@echo "Training with custom config..."
	@echo "Usage: make train-custom CONFIG=path/to/config.yaml EPOCHS=10"
	python3 train.py \
		--config $(CONFIG) \
		--overrides \
			training.epochs=$(EPOCHS) \
			data.root_dir=./dummy_hyspecnet11k

eval: 
	@echo "Evaluating TCN model..."
	@if [ -f ./checkpoints/best_val_loss.pt ]; then \
		python3 evaluate.py \
			--config hsi_compression/configs/models/tcn_lossless.yaml \
			--checkpoint ./checkpoints/best_val_loss.pt \
			--output ./results_tcn/ \
			--overrides data.root_dir=./dummy_hyspecnet11k; \
		echo "Results saved to ./results_tcn/results.json"; \
		cat ./results_tcn/results.json; \
	else \
		echo "No checkpoint found. Run 'make train-tcn' first."; \
	fi

eval-mamba:
	@echo "Evaluating Mamba model..."
	@if [ -f ./checkpoints/best_val_loss.pt ]; then \
		python3 evaluate.py \
			--config hsi_compression/configs/models/mamba_lossy.yaml \
			--checkpoint ./checkpoints/best_val_loss.pt \
			--output ./results_mamba/ \
			--overrides data.root_dir=./dummy_hyspecnet11k; \
		echo "Results saved to ./results_mamba/results.json"; \
		cat ./results_mamba/results.json; \
	else \
		echo "No checkpoint found. Run 'make train-mamba' first."; \
	fi

eval-custom:
	@echo "Evaluating with custom checkpoint..."
	@echo "Usage: make eval-custom CHECKPOINT=path/to/checkpoint.pt"
	python3 evaluate.py \
		--config hsi_compression/configs/models/tcn_lossless.yaml \
		--checkpoint $(CHECKPOINT) \
		--output ./results_custom/

# ============================================================================
# CODE QUALITY
# ============================================================================

lint:
	@echo "🔍 Checking code style (flake8)..."
	python3 -m flake8 hsi_compression/ train.py evaluate.py test_e2e.py \
		--max-line-length=120 \
		--ignore=E203,W503,E501 \
		--statistics || true
	@echo "Lint check complete"

format:
	@echo "🎨 Formatting code (autopep8)..."
	python3 -m autopep8 --in-place --aggressive --aggressive \
		-r hsi_compression/ train.py evaluate.py test_e2e.py
	@echo "Code formatted"

type-check:
	@echo "🔎 Type checking (mypy)..."
	python3 -m mypy hsi_compression/ --ignore-missing-imports --follow-imports=skip || true
	@echo "Type check complete"

clean:
	@echo "Cleaning up..."
	rm -rf checkpoints/ results_tcn/ results_mamba/ results_custom/ __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cleanup complete"

clean-all: clean
	@echo "Complete cleanup (including data)..."
	rm -rf dummy_hyspecnet11k/ test_output.log
	@echo "All cleanup complete"

clean-cache:
	@echo "Removing Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache cleaned"

# ============================================================================
# DEMO & WORKFLOWS
# ============================================================================

setup-quick: install data
	@echo "Quick setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  make test-quick     - Run sanity checks"
	@echo "  make test           - Run full E2E tests"
	@echo "  make train-tcn      - Train TCN model"
	@echo "  make eval           - Evaluate model"

demo: setup test train-tcn eval
	@echo ""
	@echo "Full demo complete!"
	@echo ""
	@echo "You've successfully:"
	@echo "  - Set up the environment"
	@echo "  - Generated test data"
	@echo "  - Ran E2E tests"
	@echo "  - Trained the TCN model"
	@echo "  - Evaluated the model"
	@echo ""
	@echo "Checkpoints: ./checkpoints/"
	@echo "Results: ./results_tcn/results.json"