#!/bin/bash
# RunPod Setup Script for CatalanSTT Training
# Run this on a RunPod GPU instance

set -e

echo "=== CatalanSTT RunPod Setup ==="

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers datasets librosa jiwer evaluate tensorboard elevenlabs

# Clone repo (if not already present)
if [ ! -d "catalan-stt" ]; then
    echo "Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/catalan-stt.git
    cd catalan-stt
else
    cd catalan-stt
    git pull
fi

# Download datasets
echo "Downloading datasets (~2GB)..."
python scripts/download_datasets.py --dataset quick

# Prepare data
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
echo "python scripts/train.py \\"
echo "  --config configs/default.yaml \\"
echo "  --train-manifest data/splits/train.json \\"
echo "  --val-manifest data/splits/val.json"
echo ""
echo "Estimated time: 20-30 minutes on RTX 3090/4090"
