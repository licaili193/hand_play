import os
import json
import numpy as np
import pretty_midi
import soundfile as sf
import torch
import copy

# Paths (adjust these to your setup)
MIDI_PATH = "examples\\example_2.mid"            # Input MIDI file
AUDIO_PATH = "examples\\example_2.wav"           # Temporary audio file path
CHECKPOINT_PATH = "PianoMotion10M\\checkpoints\\diffusion_posiguide_hubertlarge_tf2\\piano2pose-iter=90000-val_loss=0.0364401508122683.ckpt"  # Downloaded model checkpoint
OUTPUT_JSON = "predicted_hand_motion.json"      # Output JSON path

# 1. Convert MIDI to audio (16 kHz mono) using pretty_midi
print("Converting MIDI to audio...")
pm = pretty_midi.PrettyMIDI(MIDI_PATH)
# Synthesize using a sine wave for each instrument
audio_wave = pm.synthesize(fs=16000, wave=np.sin)  # Using sine wave synthesis for simplicity

# Normalize audio
audio_wave = audio_wave.astype(np.float32)
max_val = np.max(np.abs(audio_wave))
if max_val > 0:
    audio_wave /= max_val  # normalize to [-1, 1]

# Save audio (optional, for inspection)
sf.write(AUDIO_PATH, audio_wave, samplerate=16000)

# 2. Load PianoMotion10M model with proper architecture
print("Loading PianoMotion10M model checkpoint...")
import sys
sys.path.append('PianoMotion10M')  # Add PianoMotion10M to Python path

from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D

# Create args objects with the necessary parameters based on the checkpoint
class Piano2PosiArgs:
    def __init__(self):
        self.feature_dim = 512
        self.bs_dim = 6  # Position predictor uses 6 (xyz coordinates for both hands)
        self.loss_mode = 'naive_l2'
        self.encoder_type = 'transformer'
        self.hidden_type = 'audio_f'
        self.wav2vec_path = "facebook/hubert-large-ls960-ft"
        self.max_seq_len = 1500
        self.period = 1500
        self.num_layer = 8  # Large model uses 8 layers
        self.latest_layer = "tanh"

class DiffusionArgs:
    def __init__(self):
        self.unet_dim = 256  # Large model uses 256
        self.timesteps = 1000
        self.train_sec = 4  # Default training uses 4 seconds
        self.xyz_guide = True
        self.remap_noise = True
        self.bs_dim = 96  # Gesture generator uses 96 (32 joints * 3 coordinates)
        self.encoder_type = 'transformer'
        self.num_layer = 8
        self.experiment_name = "midi_inference"

# Create model components
piano2posi_args = Piano2PosiArgs()
diffusion_args = DiffusionArgs()

# Create Piano2Posi model
piano2posi = Piano2Posi(piano2posi_args)

# Create Unet1D model
cond_dim = 1024  # for hubert-large
unet = Unet1D(
    dim=diffusion_args.unet_dim,
    dim_mults=(1, 2, 4, 8),
    channels=diffusion_args.bs_dim,
    remap_noise=diffusion_args.remap_noise,
    condition=True,
    guide=diffusion_args.xyz_guide,
    guide_dim=6 if diffusion_args.xyz_guide else 0,
    condition_dim=cond_dim,
    encoder_type=diffusion_args.encoder_type,
    num_layer=diffusion_args.num_layer
)

# Create the full diffusion model
model = GaussianDiffusion1D_piano2pose(
    unet,
    piano2posi,
    seq_length=diffusion_args.train_sec * 30,
    timesteps=diffusion_args.timesteps,
    objective='pred_v',
)

# Load the checkpoint
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)['state_dict']
model.load_state_dict(state_dict)
model.eval()

# Force CPU usage to avoid CUDA compatibility issues
device = torch.device('cpu')
model.to(device)
print(f"Model loaded on device: {device}")

# 3. Run inference
print("Running inference on audio...")
# Prepare audio input
audio_tensor = torch.tensor(audio_wave).unsqueeze(0).to(device)  # add batch dimension

# Use the model's expected sequence length
seq_length = diffusion_args.train_sec * 30  # 10 seconds * 30 FPS = 300 frames
frame_num = seq_length

print(f"Audio length: {len(audio_wave)} samples ({len(audio_wave)/16000:.2f} seconds)")
print(f"Using sequence length: {seq_length} frames ({diffusion_args.train_sec} seconds)")

# Create dummy pose tensors for the guide (required by the model)
# The model expects guide poses for both hands
dummy_guide = torch.zeros(1, frame_num, 6).to(device)  # 6 = 3 coordinates for each hand

with torch.no_grad():
    # Sample from the diffusion model
    pose_hat, guide = model.sample(audio_tensor, frame_num, 1)
    pose_hat = pose_hat.permute(0, 2, 1)  # Rearrange dimensions
    guide = guide.permute(0, 2, 1)

# Convert to numpy and process
prediction = pose_hat[0].detach().cpu().numpy() * np.pi  # Convert to radians
guide = guide[0].cpu().numpy()

# Apply smoothing
from scipy.signal import savgol_filter
for i in range(prediction.shape[1]):
    prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
for i in range(guide.shape[1]):
    guide[:, i] = savgol_filter(guide[:, i], 5, 2)

# Reshape output into (Frames, J, 3)
# The model outputs 96 dimensions (32 joints * 3 coordinates)
num_coords = prediction.shape[1]
J = num_coords // 3
pred_joints = prediction.reshape((-1, J, 3))
num_frames = pred_joints.shape[0]
print(f"Predicted {J} joints per frame, total frames: {num_frames}")

# Split into left and right hand (assuming first half of joints belong to one hand and second half to the other)
left_hand_joints = pred_joints[:, :int(J/2), :].tolist()
right_hand_joints = pred_joints[:, int(J/2):, :].tolist()

# 4. Save output to JSON
output_data = {"frames": []}
for i in range(num_frames):
    frame_data = {
        "frame_index": i,
        "left_hand_joints": left_hand_joints[i],
        "right_hand_joints": right_hand_joints[i]
    }
    output_data["frames"].append(frame_data)

# Add metadata about joint structure
output_data["metadata"] = {
    "total_joints_per_hand": J//2,
    "total_frames": num_frames,
    "joint_structure": "16 joints per hand (MANO)",
    "coordinate_system": "3D Cartesian (X, Y, Z)",
    "hand_split": "First 16 joints = left hand, Last 16 joints = right hand",
    "model_type": "Diffusion model with Piano2Posi + Unet1D",
    "checkpoint_loaded": True
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"Saved predicted motion to {OUTPUT_JSON}")

# 5. (Optional) Simple Visualization of one frame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Import MANO utilities
from mano_utils import (
    get_mano_joint_connections,
    get_finger_colors,
    get_finger_ranges,
    save_joint_mapping_info,
    visualize_both_hands_3d
)

# Get anatomically precise MANO connections
mano_connections = get_mano_joint_connections()

# Save joint mapping information for reference
save_joint_mapping_info(J, J//2)

# Visualize both hands using the utility function
visualize_both_hands_3d(left_hand_joints[-1], right_hand_joints[-1], num_frames-1, num_frames)
