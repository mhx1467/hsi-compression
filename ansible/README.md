Step 1: Install Ansible
pip install ansible>=2.9
cd ansible
Step 2: Configure Machines
Copy the example inventory:

cp inventory.example.ini inventory.ini
Edit inventory.ini with your machine details:

[gpu_servers]
my-gpu-1 ansible_host=192.168.1.100 ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/my-key.pem
my-gpu-2 ansible_host=192.168.1.101 ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/my-key.pem
Step 3: Test SSH Connection
./quickstart.sh test
Or manually:

ansible all -i inventory.ini -m ping
Step 4: Setup Remote Machines
Setup all GPU servers:

./quickstart.sh setup
Or setup a specific machine:

./quickstart.sh setup gpu-server-1
Or run manually:

ansible-playbook setup.yml -i inventory.ini -l gpu_servers
Step 5: Prepare Dataset
./quickstart.sh dataset
Or manually:

ansible-playbook dataset.yml -i inventory.ini -l gpu_servers
Step 6: Train Model
./quickstart.sh train -e training_epochs=100
Or manually:

ansible-playbook train.yml -i inventory.ini -l gpu_servers \
  -e "model_name=tcn_lossless" \
  -e "training_epochs=100"
Step 7: Evaluate Model
./quickstart.sh eval
Or manually:

ansible-playbook evaluate.yml -i inventory.ini -l gpu_servers \
  -e "checkpoint_path=/home/ubuntu/hsi-compression/checkpoints/best_val_loss.pt"
Advanced Usage
Training Multiple Models in Parallel
# Train different models on different servers
ansible-playbook train.yml -i inventory.ini -l gpu-server-1 \
  -e "model_name=tcn_lossless" -e "training_epochs=100" &

ansible-playbook train.yml -i inventory.ini -l gpu-server-2 \
  -e "model_name=mamba_lossy" -e "training_epochs=100" &

wait
Asynchronous Training
# Start training on background
ansible-playbook train.yml -i inventory.ini -l gpu_servers \
  -e "model_name=tcn_lossless" \
  -e "training_epochs=500" \
  -e "run_async=true"

# Training continues in background, you can check logs via SSH
ssh ubuntu@192.168.1.100
tail -f ~/hsi-compression/logs/training_*.log
Training with WandB Integration
./quickstart.sh train \
  -e "wandb_enabled=true" \
  -e "wandb_api_key=your-api-key-here" \
  -e "wandb_project=hsi-compression-exp1"
Or edit host_vars/gpu-server-1.yml:

---
wandb_enabled: true
wandb_api_key: "your-secret-key"
wandb_project: "hsi-compression"
Custom Training Parameters
./quickstart.sh train \
  -e "model_name=tcn_lossless" \
  -e "training_epochs=200" \
  -e "batch_size=16" \
  -e "learning_rate=0.0002"
Run Everything at Once
./quickstart.sh all \
  -e "model_name=tcn_lossless" \
  -e "training_epochs=100"
This runs:

System setup
Python setup
Project setup
Dataset setup
Training
Evaluation
Monitoring and Debugging
View Training Progress in Real-Time
# SSH to the machine
ssh ubuntu@192.168.1.100

# Check active process
ps aux | grep python

# View logs in real-time
tail -f ~/hsi-compression/logs/training_*.log

# Check GPU utilization
nvidia-smi watch -n 1
View Checkpoints on Remote
ansible gpu_servers -i inventory.ini -m shell \
  -a "ls -lh ~/hsi-compression/checkpoints/"
Download Results Locally
mkdir -p results
scp -r ubuntu@192.168.1.100:~/hsi-compression/results/* results/
scp -r ubuntu@192.168.1.100:~/hsi-compression/logs/* results/logs/
Check for Errors
# View Ansible logs
cat ansible.log

# SSH to machine and check error logs
ssh ubuntu@192.168.1.100
tail -f ~/hsi-compression/logs/training_*.log | grep -i error
Run playbook in check mode (dry-run)
ansible-playbook train.yml -i inventory.ini -l gpu_servers --check