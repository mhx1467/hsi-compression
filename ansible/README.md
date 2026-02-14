# Ansible Setup and Training Guide - Simplified Version

## Overview

This simplified setup assumes **CUDA is already installed** on your remote machine and uses **pip** to install PyTorch and mamba-ssm into the existing Python virtual environment.

## Quick Start

### Step 1: Install Ansible

```bash
pip install ansible>=2.9
cd ansible
```

### Step 2: Configure Your Remote Machine

Copy and edit the inventory:

```bash
cp inventory.example.ini inventory.ini
```

Edit `inventory.ini`:

```ini
[gpu_servers]
gpu-server-1 ansible_host=142.170.89.112 ansible_user=root ansible_port=32244
```

**Important:** Your remote machine must have:
- CUDA 11.6+ already installed
- Python 3.10+ available
- GPU with nvidia-smi working

### Step 3: Test SSH Connection

```bash
./quickstart.sh test
```

Or manually:

```bash
ansible all -i inventory.ini -m ping
```

### Step 4: Setup Remote Machine

```bash
./quickstart.sh setup
```

This will:
1. Check CUDA is available
2. Install system dependencies
3. Clone/setup the project
4. Create Python virtual environment (Python 3.10)
5. Install PyTorch with CUDA 12.1 wheels into venv
6. Install mamba-ssm with `--no-build-isolation` into venv

### Step 5: Verify Installation

```bash
./quickstart.sh verify
```

This checks:
- CUDA availability
- Virtual environment
- PyTorch installation
- mamba-ssm import

## Manual Installation (if Ansible fails)

If you prefer to install manually on the remote machine:

```bash
# SSH to remote
ssh root@142.170.89.112 -p 32244

# Navigate to project
cd /home/root/hsi-compression

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install build tools
pip install ninja packaging

# Install mamba-ssm
pip install mamba-ssm --no-build-isolation

# Verify
python -c "from mamba_ssm import Mamba; print('Success')"
```

## Training

### Activate Environment

```bash
# SSH to remote
ssh root@142.170.89.112 -p 32244

# Activate virtual environment
source /home/root/hsi-compression/venv/bin/activate
```

### Train TCN Model

```bash
python train.py --config hsi_compression/configs/models/tcn_lossless.yaml
```

### Train Mamba Model

```bash
python train.py --config hsi_compression/configs/models/mamba_lossy.yaml
```

## Troubleshooting

### Issue: CUDA not found

Check if CUDA is installed on remote:

```bash
ssh root@your-machine
nvidia-smi
nvcc --version
```

If not installed, install CUDA manually or contact your cloud provider.

### Issue: Virtual environment not found

The virtual environment is created in: `{{ project_dir }}/venv`

If it doesn't exist, ensure Python setup completed:

```bash
ssh root@your-machine
ls -la /home/root/hsi-compression/venv
```

### Issue: mamba-ssm import fails

Check PyTorch installation:

```bash
source /home/root/hsi-compression/venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Ensure you have:
- Python 3.10
- PyTorch with CUDA 12.1 support
- ninja and packaging installed

Then reinstall:

```bash
pip install mamba-ssm --no-build-isolation --force-reinstall
```

### Issue: PyTorch CUDA mismatch

If you see CUDA version mismatch errors, ensure you're using the correct wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

## Project Structure

```
/home/root/hsi-compression/
├── venv/                       # Virtual environment (Python 3.10)
├── hsi_compression/            # Main package
│   ├── models/                # TCN and Mamba models
│   ├── configs/               # Model configs
│   └── ...
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
└── checkpoints/               # Saved models
```

## Virtual Environment

The virtual environment is created at: `{{ venv_dir }}` (default: `/home/root/hsi-compression/venv`)

It includes:
- Python 3.10
- PyTorch (with CUDA 12.1)
- mamba-ssm
- All other dependencies from pyproject.toml

Activate with:

```bash
source {{ venv_dir }}/bin/activate
```

## Next Steps

1. **Pull Dataset**
   ```bash
   ansible-playbook dataset.yml -i inventory.ini -l gpu_servers
   ```

2. **Run Training**
   ```bash
   source {{ venv_dir }}/bin/activate
   python train.py --config hsi_compression/configs/models/tcn_lossless.yaml
   ```

3. **Evaluate Model**
   ```bash
   source {{ venv_dir }}/bin/activate
   python evaluate.py --checkpoint checkpoints/best_model.pt
   ```

## References

- Mamba-SSM: https://github.com/state-spaces/mamba
- PyTorch CUDA Wheels: https://download.pytorch.org/whl/torch_stable.html

