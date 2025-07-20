# 1. Clone the PianoMotion10M repository
git clone https://github.com/agnJason/PianoMotion10M.git
cd PianoMotion10M

# 2. Create a Python virtual environment (using venv here; alternatively use Conda)
python -m venv venv
Set-ExecutionPolicy Bypass -Scope Process -Force; .\venv\Scripts\Activate.ps1  # Activate venv in PowerShell

# 3. Install core dependencies for inference (avoid training/rendering-specific packages)
pip install --upgrade pip

# Install PyTorch (CPU-only to simplify; for GPU, install the CUDA-enabled wheel as appropriate)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Hugging Face Transformers (for loading the HuBERT audio model) and other needed libs
pip install transformers numpy scipy librosa pretty_midi matplotlib pytorch-lightning einops smplx
 
# Note: 'pretty_midi' and 'librosa' are for MIDI-to-audio conversion and audio processing.
# We do NOT install mamba-ssm or nvdiffrast to avoid Windows issues.

# 4. (Optional) Download pre-trained model checkpoints from Hugging Face
# We will use the HuBERT-large Transformer model checkpoint (no mamba).
# For convenience, use the huggingface_hub Python library to download the checkpoint.
pip install huggingface_hub  # to use huggingface_hub Python library

# Create checkpoints directory if it doesn't exist
if (!(Test-Path ".\checkpoints")) {
    New-Item -ItemType Directory -Path ".\checkpoints"
}

# Download the model checkpoint using Python
python -c "
from huggingface_hub import hf_hub_download
import os

# Download the checkpoint file
checkpoint_path = hf_hub_download(
    repo_id='agnJason/PianoMotion_models',
    filename='diffusion_posiguide_hubertlarge_tf2/piano2pose-iter=90000-val_loss=0.0364401508122683.ckpt',
    local_dir='./checkpoints'
)
print(f'Downloaded checkpoint to: {checkpoint_path}')
"

# Alternatively, download the file from the HuggingFace page and place it under PianoMotion10M\checkpoints.
