#!/bin/bash
# Script to install PyTorch with CUDA 11.8 support
# This should be run after installing requirements.txt

echo "Installing PyTorch 2.1.0 with CUDA 11.8 support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "PyTorch installation complete!"
echo "Verify installation with: python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"

