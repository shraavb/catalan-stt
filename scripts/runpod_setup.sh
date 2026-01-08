#!/bin/bash
# RunPod Setup Script for CatalanSTT Training
# Run this on a RunPod GPU instance

set -e

echo "=== CatalanSTT RunPod Setup ==="

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers datasets librosa jiwer evaluate tensorboard accelerate huggingface_hub tqdm requests

# Clone repo (if not already present)
if [ ! -d "catalan-stt" ]; then
    echo "Cloning repository..."
    git clone https://github.com/shraavb/catalan-stt.git
    cd catalan-stt
else
    cd catalan-stt
    git pull
fi

# Set PYTHONPATH so 'src' module is found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create data directories
mkdir -p data/raw/archives

# Download datasets from Hugging Face (~1.8GB)
echo ""
echo "Downloading datasets from Hugging Face (~1.8GB)..."
python -c "
from huggingface_hub import hf_hub_download
import os

os.makedirs('data/raw/archives', exist_ok=True)

print('Downloading ca_es_female.zip...')
hf_hub_download(
    repo_id='shraavb/catalan-stt-data',
    filename='ca_es_female.zip',
    repo_type='dataset',
    local_dir='data/raw/archives',
    local_dir_use_symlinks=False
)

print('Downloading ca_es_male.zip...')
hf_hub_download(
    repo_id='shraavb/catalan-stt-data',
    filename='ca_es_male.zip',
    repo_type='dataset',
    local_dir='data/raw/archives',
    local_dir_use_symlinks=False
)

print('Downloads complete!')
"

# Extract datasets
echo ""
echo "Extracting datasets..."
mkdir -p data/raw/google-catalan-female data/raw/google-catalan-male
unzip -q -o data/raw/archives/ca_es_female.zip -d data/raw/google-catalan-female
unzip -q -o data/raw/archives/ca_es_male.zip -d data/raw/google-catalan-male
echo "Extraction complete!"

# Prepare data
echo ""
echo "Preparing datasets..."
python scripts/prepare_datasets.py --input-dir data/raw --output-dir data/processed

# Check GPU
echo ""
echo "=== GPU Status ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Ready to Train ==="
echo "Run the following command to start training:"
echo ""
echo "export PYTHONPATH=\${PYTHONPATH}:\$(pwd)"
echo "python scripts/train.py \\"
echo "  --config configs/default.yaml \\"
echo "  --train-manifest data/splits/train.json \\"
echo "  --val-manifest data/splits/val.json"
echo ""
echo "Estimated time: 20-30 minutes on RTX 3090/4090"
