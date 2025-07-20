import os
import json
import numpy as np
import pretty_midi
import soundfile as sf
import torch

# Paths (adjust these to your setup)
MIDI_PATH = "examples\\example_1.mid"            # Input MIDI file
AUDIO_PATH = "examples\\example_1.wav"           # Temporary audio file path
CHECKPOINT_PATH = "PianoMotion10M\\checkpoints\\diffusion_posiguide_hubertlarge_tf2\\piano2pose-iter=90000-val_loss=0.0364401508122683.ckpt"  # Downloaded model checkpoint
OUTPUT_JSON = "predicted_hand_motion.json"      # Output JSON path

# 1. Convert MIDI to audio (16 kHz mono) using pretty_midi
print("Converting MIDI to audio...")
pm = pretty_midi.PrettyMIDI(MIDI_PATH)
# Synthesize using a sine wave for each instrument
audio_wave = pm.synthesize(fs=16000, wave=np.sin)  # Using sine wave synthesis for simplicity
# If pretty_midi doesn't have built-in synth, alternatively use fluidsynth as discussed.

# Normalize audio
audio_wave = audio_wave.astype(np.float32)
max_val = np.max(np.abs(audio_wave))
if max_val > 0:
    audio_wave /= max_val  # normalize to [-1, 1]

# Save audio (optional, for inspection)
sf.write(AUDIO_PATH, audio_wave, samplerate=16000)

# 2. Load pre-trained audio feature extractor (HuBERT)
print("Loading HuBERT model for audio feature extraction...")
from transformers import Wav2Vec2Processor, Wav2Vec2Model
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
audio_model = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft")
audio_model.eval()

# Prepare input for HuBERT
inputs = processor(audio_wave, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    audio_features = audio_model(**inputs).last_hidden_state  # shape: (1, time_steps, feature_dim)
audio_features = audio_features[0]  # remove batch dim, shape: (T_audio, 1024) for hubert-large

print(f"Extracted audio features shape: {audio_features.shape}")

# 3. Load PianoMotion10M model (Position Predictor + Gesture Generator)
print("Loading PianoMotion10M model checkpoint...")
import sys
sys.path.append('PianoMotion10M')  # Add PianoMotion10M to Python path

from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D

# Load the model components as shown in the original infer.py
# We'll need to set up the model architecture similar to the original code
# For now, let's create a simplified version that loads the checkpoint

# Create a simple args object with the necessary parameters
class Args:
    def __init__(self):
        self.feature_dim = 512
        self.bs_dim = 96  # 32 joints * 3 coordinates
        self.loss_mode = 'naive_l2'
        self.encoder_type = 'transformer'
        self.hidden_type = 'audio_f'
        self.wav2vec_path = "facebook/hubert-large-ls960-ft"
        self.max_seq_len = 1500  # Increased to accommodate longer sequences
        self.period = 1500
        self.num_layer = 4
        self.latest_layer = "tanh"

# Create model components
args = Args()
piano2posi = Piano2Posi(args)

# For now, let's use a simplified approach - just use the piano2posi model directly
# since the full diffusion model requires more complex setup
model = piano2posi
model.eval()

# 4. Run inference
print("Running inference on audio features...")
# Convert audio features to the format expected by piano2posi
# The model expects raw audio, not pre-extracted features
audio_tensor = torch.tensor(audio_wave).unsqueeze(0)  # add batch dimension
frame_num = audio_features.shape[0]  # number of time steps

with torch.no_grad():
    pred_joint_seq = model(audio_tensor, frame_num)  # forward pass; shape (B, Frames, J*3)
pred_joint_seq = pred_joint_seq[0].cpu().numpy()  # shape (Frames, J*3)

# Reshape output into (Frames, J, 3)
# We'll determine J from the model or config. If bs_dim=96 in config, J*3 = 96 -> J=32 joints total.
num_coords = pred_joint_seq.shape[-1]
J = num_coords // 3
pred_joints = pred_joint_seq.reshape((-1, J, 3))
num_frames = pred_joints.shape[0]
print(f"Predicted {J} joints per frame, total frames: {num_frames}")

# Split into left and right hand (assuming first half of joints belong to one hand and second half to the other).
# The MANO model typically provides joints in a specific order; documentation from dataset may clarify.
# For now, assume joints [0:16] = left hand, [16:32] = right hand (or vice versa).
left_hand_joints = pred_joints[:, :int(J/2), :].tolist()
right_hand_joints = pred_joints[:, int(J/2):, :].tolist()

# 5. Save output to JSON
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
    "joint_structure": "16 joints per hand (simplified MANO)",
    "coordinate_system": "3D Cartesian (X, Y, Z)",
    "hand_split": "First 16 joints = left hand, Last 16 joints = right hand"
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"Saved predicted motion to {OUTPUT_JSON}")

# 6. (Optional) Simple Visualization of one frame
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
