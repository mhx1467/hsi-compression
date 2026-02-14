#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}HSI Compression Framework - Setup${NC}"
echo -e "${BLUE}============================================${NC}\n"

print_section() {
    echo -e "\n${YELLOW}==> $1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Ansible is installed
print_section "Checking Ansible installation"
if ! command -v ansible &> /dev/null; then
    print_error "Ansible is not installed"
    echo "Install it with: pip install ansible>=2.9"
    exit 1
fi
print_success "Ansible $(ansible --version | head -1)"

# Check inventory file
print_section "Checking inventory configuration"
if [ ! -f "inventory.ini" ]; then
    print_error "inventory.ini not found"
    echo "Creating from example..."
    if [ -f "inventory.example.ini" ]; then
        cp inventory.example.ini inventory.ini
        print_success "Created inventory.ini from template"
        echo -e "\n${YELLOW}Please edit inventory.ini and add your GPU server credentials${NC}"
    else
        print_error "inventory.example.ini not found"
        exit 1
    fi
fi
print_success "inventory.ini exists"

# Parse command
COMMAND="${1:-help}"

case $COMMAND in
    help)
        echo -e "\n${BLUE}Usage: ./quickstart.sh [command] [options]${NC}\n"
        echo "Commands:"
        echo "  help              Show this help message"
        echo "  test              Test connection to all hosts"
        echo "  setup             Setup all GPU servers"
        echo "  setup [host]      Setup specific host"
        echo "  dataset           Pull dataset on all servers"
        echo "  train [options]   Train model on all servers"
        echo "  eval [options]    Evaluate model on all servers"
        echo "  all [options]     Run complete workflow"
        echo ""
        echo "Examples:"
        echo "  ./quickstart.sh test                    # Test SSH connections"
        echo "  ./quickstart.sh setup gpu-server-1      # Setup one machine"
        echo "  ./quickstart.sh train -e training_epochs=200  # Train for 200 epochs"
        echo ""
        ;;
    
    test)
        print_section "Testing connection to all hosts"
        ansible all -i inventory.ini -m ping
        print_success "Connection test completed"
        ;;
    
    setup)
        HOST="${2:-gpu_servers}"
        print_section "Setting up $HOST"
        echo -e "${YELLOW}This will install system dependencies, Python, and the project.${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ansible-playbook setup.yml -i inventory.ini -l "$HOST"
            print_success "Setup completed for $HOST"
        else
            echo "Cancelled"
        fi
        ;;
    
    dataset)
        print_section "Setting up dataset on gpu_servers"
        ansible-playbook dataset.yml -i inventory.ini -l gpu_servers
        print_success "Dataset setup completed"
        ;;
    
    train)
        print_section "Starting training on gpu_servers"
        shift
        EXTRA_ARGS="$@"
        
        # Set defaults if not provided
        if [[ ! "$EXTRA_ARGS" =~ "model_name" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS -e model_name=tcn_lossless"
        fi
        if [[ ! "$EXTRA_ARGS" =~ "training_epochs" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS -e training_epochs=100"
        fi
        
        ansible-playbook train.yml -i inventory.ini -l gpu_servers $EXTRA_ARGS
        print_success "Training started"
        ;;
    
    eval)
        print_section "Starting evaluation on gpu_servers"
        shift
        EXTRA_ARGS="$@"
        
        # Set default checkpoint if not provided
        if [[ ! "$EXTRA_ARGS" =~ "checkpoint_path" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS -e 'checkpoint_path={{ checkpoint_dir }}/best_val_loss.pt'"
        fi
        
        ansible-playbook evaluate.yml -i inventory.ini -l gpu_servers $EXTRA_ARGS
        print_success "Evaluation started"
        ;;
    
    all)
        print_section "Running complete workflow"
        shift
        EXTRA_ARGS="$@"
        
        # Set defaults if not provided
        if [[ ! "$EXTRA_ARGS" =~ "model_name" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS -e model_name=tcn_lossless"
        fi
        if [[ ! "$EXTRA_ARGS" =~ "training_epochs" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS -e training_epochs=100"
        fi
        
        echo -e "${YELLOW}This will run: setup → dataset → train → eval${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ansible-playbook all.yml -i inventory.ini -l gpu_servers $EXTRA_ARGS
            print_success "Complete workflow finished"
        else
            echo "Cancelled"
        fi
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        echo "Run: ./quickstart.sh help"
        exit 1
        ;;
esac

echo ""
