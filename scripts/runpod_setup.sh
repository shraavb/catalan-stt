#!/bin/bash
# RunPod Setup Script for SpanishSlangSTT Training
# Run this on a RunPod GPU instance

set -e

echo "=== SpanishSlangSTT RunPod Setup ==="

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers datasets librosa jiwer evaluate tensorboard accelerate huggingface_hub tqdm requests

# Clone repo (if not already present)
if [ ! -d "spanish-slang-stt" ]; then
    echo "Cloning repository..."
    git clone https://github.com/shraavb/spanish-slang-stt.git
    cd spanish-slang-stt
else
    cd spanish-slang-stt
    git pull
fi

# Set PYTHONPATH so 'src' module is found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create data directories
mkdir -p data/raw data/processed data/splits

# Check GPU
echo ""
echo "=== GPU Status ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Ready to Train ==="
echo "Place your training data in data/raw/ then run:"
echo ""
echo "export PYTHONPATH=\${PYTHONPATH}:\$(pwd)"
echo "python scripts/prepare_datasets.py --input-dir data/raw --output-dir data/processed"
echo ""
echo "Then start training with:"
echo "python scripts/train.py \\"
echo "  --config configs/default.yaml \\"
echo "  --train-manifest data/splits/train.json \\"
echo "  --val-manifest data/splits/val.json"
echo ""
echo "Estimated time: 20-30 minutes on RTX 3090/4090"
