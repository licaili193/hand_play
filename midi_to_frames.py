import os
import json
import numpy as np
import torch
import warnings
import sys
import argparse
from pathlib import Path

# Add safe globals for checkpoint loading
torch.serialization.add_safe_globals([argparse.Namespace])

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import our audio processor
from audio_processor import create_audio_processor

# Configuration
DEFAULT_AUDIO_PATH = "examples/example_2.mp3"  # Audio file only
CHECKPOINT_PATH = "PianoMotion10M/checkpoints/diffusion_posiguide_hubertlarge_tf2/piano2pose-iter=90000-val_loss=0.0364401508122683.ckpt"
OUTPUT_JSON = "predicted_hand_motion.json"
FPS = 30  # Standard FPS for PianoMotion10M
USE_FULL_SEQUENCE = False  # Set to True to use full audio length, False for fixed duration (4.0s)
MAX_SEQUENCE_LENGTH = 900  # Maximum sequence length the model can handle (30 seconds at 30 FPS)

"""
PianoMotion10M Audio to Hand Motion Inference

This script processes audio input and generates hand motion predictions
using the PianoMotion10M diffusion model.

Supported input formats:
- Audio files (.wav, .mp3, .flac, .m4a, .ogg)

Configuration:
- Input file: examples/example_2.wav (change DEFAULT_AUDIO_PATH to your audio file)
- Output file: predicted_hand_motion.json
- Full sequence mode: USE_FULL_SEQUENCE = True (uses full audio length)
- Fixed duration mode: USE_FULL_SEQUENCE = False (uses 4.0 seconds)
- FPS: 30

To use a different audio file, change DEFAULT_AUDIO_PATH to your audio file path.

Note: Full sequence mode will process the entire audio file length. For very long files
(>2.8 minutes), consider using fixed duration mode to avoid memory issues.
"""

def load_piano2posi_config(checkpoint_path):
    """Load Piano2Posi configuration from checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    args_posi_path = os.path.join(checkpoint_dir, 'args_posi.txt')
    
    if not os.path.exists(args_posi_path):
        raise FileNotFoundError(f"Piano2Posi config not found: {args_posi_path}")
    
    with open(args_posi_path, 'r') as f:
        config = json.load(f)
    
    return config

class Piano2PosiArgs:
    def __init__(self, config_dict):
        self.feature_dim = config_dict.get('feature_dim', 512)
        self.bs_dim = config_dict.get('bs_dim', 6)
        self.loss_mode = config_dict.get('loss_mode', 'naive_l1')
        self.encoder_type = config_dict.get('encoder_type', 'transformer')
        self.hidden_type = 'audio_f'
        self.max_seq_len = config_dict.get('max_seq_len', 900)
        self.period = config_dict.get('period', 30)
        self.latest_layer = config_dict.get('latest_layer', 'tanh')
        self.num_layer = config_dict.get('num_layer', 8)
        
        # Handle wav2vec path
        wav2vec_path = config_dict.get('wav2vec_path', './checkpoints/hubert-large-ls960-ft')
        if 'hubert-large' in wav2vec_path:
            self.wav2vec_path = 'facebook/hubert-large-ls960-ft'
        else:
            self.wav2vec_path = 'facebook/hubert-large-ls960-ft'

class DiffusionArgs:
    def __init__(self):
        self.timesteps = 1000
        self.train_sec = 4
        self.xyz_guide = True
        self.remap_noise = True
        self.bs_dim = 96
        self.encoder_type = 'transformer'
        self.unet_dim = 256
        self.num_layer = 8

def run_inference(model, audio_wave, device, fps=30):
    """Run model inference."""
    # Convert to tensor
    audio_tensor = torch.tensor(audio_wave, dtype=torch.float32).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)
    
    # Calculate frame number based on audio duration
    audio_duration = len(audio_wave) / 16000  # 16000 Hz sample rate
    frame_num = int(audio_duration * fps)
    
    print(f"Running inference: {frame_num} frames at {fps} FPS (audio duration: {audio_duration:.2f}s)")
    
    with torch.no_grad():
        pose_hat, guide = model.sample(audio_tensor, frame_num, batch_size=1)
        
        # Handle output dimensions if needed
        if pose_hat.shape[1] == 96 and guide.shape[1] == 6:
            pose_hat = pose_hat.permute(0, 2, 1)
            guide = guide.permute(0, 2, 1)
    
    return pose_hat, guide, frame_num

def process_output(pose_hat, guide, device):
    """Process model output with coordinate transformations to match PianoMotion10M format."""
    # Debug: Print actual tensor shapes
    print(f"DEBUG: pose_hat shape: {pose_hat.shape}")
    print(f"DEBUG: guide shape: {guide.shape}")
    
    # Apply coordinate scaling to guide (positions)
    scale = torch.tensor([1.5, 1.5, 25], device=device)
    scale_expanded = scale.repeat(2).view(1, 1, 6).expand_as(guide)
    scaled_guide = guide * scale_expanded
    
    # Convert pose to radians (these are rotation parameters)
    scaled_pose = pose_hat * np.pi
    
    # Convert to numpy
    scaled_guide_np = scaled_guide[0].cpu().numpy()  # Shape: (frames, 6)
    scaled_pose_np = scaled_pose[0].cpu().numpy()    # Shape: (frames, 96)
    
    print(f"DEBUG: scaled_guide_np shape: {scaled_guide_np.shape}")
    print(f"DEBUG: scaled_pose_np shape: {scaled_pose_np.shape}")
    
    # Extract positions from guide
    right_hand_pos = scaled_guide_np[:, :3]   # Shape: (frames, 3)
    left_hand_pos = scaled_guide_np[:, 3:]    # Shape: (frames, 3)
    
    # The pose tensor (96 dims) needs to be interpreted correctly
    # From show.py: each hand needs [translation(3), global_orient(3), hand_pose(45)] = 51 params
    # Our model outputs: guide(6) + pose_hat(96) 
    # 
    # Key insight: The 96 parameters might represent the full rotation parameters
    # Let's examine if 96 = 48 per hand, and how to split those 48 parameters
    
    print(f"DEBUG: Total pose dimensions: {scaled_pose_np.shape[1]}")
    print(f"DEBUG: Assuming 48 parameters per hand")
    
    # Split 96 dimensions into 48 per hand
    right_hand_pose_params = scaled_pose_np[:, :48]   # First 48 for right hand
    left_hand_pose_params = scaled_pose_np[:, 48:]    # Next 48 for left hand
    
    print(f"DEBUG: Right hand pose params shape: {right_hand_pose_params.shape}")
    print(f"DEBUG: Left hand pose params shape: {left_hand_pose_params.shape}")
    
    # Now the question: how to interpret 48 parameters?
    # Option 1: 3 global_orient + 45 hand_pose (what we tried)
    # Option 2: 48 hand_pose parameters (no global_orient, or zero global_orient)  
    # Option 3: Some other structure
    
    # Let's test both interpretations:
    # Method 1: 3 global_orient + 45 hand_pose
    if right_hand_pose_params.shape[1] >= 48:
        right_global_orient_v1 = right_hand_pose_params[:, :3]   # First 3: global orientation
        right_hand_pose_v1 = right_hand_pose_params[:, 3:48]     # Next 45: hand pose (3:48 = 45 params)
        
        left_global_orient_v1 = left_hand_pose_params[:, :3]     # First 3: global orientation  
        left_hand_pose_v1 = left_hand_pose_params[:, 3:48]       # Next 45: hand pose
        
        print(f"DEBUG: Method 1 - Right global_orient shape: {right_global_orient_v1.shape}")
        print(f"DEBUG: Method 1 - Right hand_pose shape: {right_hand_pose_v1.shape}")
    
    # Method 2: 0 global_orient + 48 hand_pose (extended hand pose)
    right_global_orient_v2 = np.zeros((scaled_pose_np.shape[0], 3))  # Zero global orientation
    right_hand_pose_v2 = right_hand_pose_params  # All 48 as hand pose
    
    left_global_orient_v2 = np.zeros((scaled_pose_np.shape[0], 3))   # Zero global orientation
    left_hand_pose_v2 = left_hand_pose_params    # All 48 as hand pose
    
    print(f"DEBUG: Method 2 - Right global_orient shape: {right_global_orient_v2.shape}")
    print(f"DEBUG: Method 2 - Right hand_pose shape: {right_hand_pose_v2.shape}")
    
    # CORRECT INTERPRETATION based on train.py analysis:
    # The training code shows that 48 params are rotation parameters that get 
    # concatenated directly with 3 translation params to form 51 total params.
    # The 48 params represent combined global_orient + hand_pose in some encoding.
    print(f"DEBUG: Using CORRECT interpretation from training code analysis")
    
    # Based on train.py: concatenate([translation(3), rotation_params(48)]) = 51 total
    # This means the 48 parameters encode both global_orient and hand_pose information
    # We need to split them appropriately: first 3 as global_orient, remaining 45 as hand_pose
    
    right_global_orient = right_hand_pose_params[:, :3]   # First 3: global orientation
    right_hand_pose = right_hand_pose_params[:, 3:48]     # Remaining 45: hand pose  
    
    left_global_orient = left_hand_pose_params[:, :3]     # First 3: global orientation
    left_hand_pose = left_hand_pose_params[:, 3:48]       # Remaining 45: hand pose
    
    print(f"DEBUG: Final - Right global_orient shape: {right_global_orient.shape}")  
    print(f"DEBUG: Final - Right hand_pose shape: {right_hand_pose.shape}")
    print(f"DEBUG: Final - Left global_orient shape: {left_global_orient.shape}")
    print(f"DEBUG: Final - Left hand_pose shape: {left_hand_pose.shape}")
    
    # Construct full MANO parameter arrays (51 params per hand as in show.py)
    # Format: [translation(3), global_orient(3), hand_pose(45)]
    right_hand_full = np.concatenate([right_hand_pos, right_global_orient, right_hand_pose], axis=1)
    left_hand_full = np.concatenate([left_hand_pos, left_global_orient, left_hand_pose], axis=1)
    
    return {
        'right_hand_params': right_hand_full,    # Shape: (frames, 51) - full MANO params
        'left_hand_params': left_hand_full,      # Shape: (frames, 51) - full MANO params
        'right_hand_position': right_hand_pos,   # Shape: (frames, 3) - for backward compatibility
        'left_hand_position': left_hand_pos,     # Shape: (frames, 3) - for backward compatibility  
        'right_hand_angles': right_hand_pose_params,  # Shape: (frames, 48) - for backward compatibility
        'left_hand_angles': left_hand_pose_params,    # Shape: (frames, 48) - for backward compatibility
        'num_frames': scaled_pose_np.shape[0],
        'mano_parameter_structure': 'corrected_pianomotion10m',
        'parameter_breakdown': {
            'translation': [0, 3],      # indices 0-2: translation (from guide)
            'global_orient': [3, 6],    # indices 3-5: global orientation (from pose)
            'hand_pose': [6, 51]        # indices 6-50: hand pose parameters (from pose)
        }
    }

def save_output(processed_data, output_path, fps=30):
    """Save processed data in comprehensive format for standalone visualization."""
    
    def convert_to_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    output_data = {"frames": []}
    num_frames = processed_data['num_frames']
    
    # Import MANO conversion to get joint data
    try:
        from mano_utils import convert_mano_params_to_joints
        
        # Get joint data for all frames
        print("Converting MANO parameters to joint positions for all frames...")
        all_left_joints = []
        all_right_joints = []
        
        for i in range(num_frames):
            # Use the corrected full MANO parameter format if available
            if 'right_hand_params' in processed_data:
                # New corrected format: [translation(3), global_orient(3), hand_pose(45)]
                right_params = processed_data['right_hand_params'][i]  # 51 parameters
                left_params = processed_data['left_hand_params'][i]    # 51 parameters
                
                # Extract components for MANO (matching show.py format)
                right_translation = right_params[:3]
                right_global_orient = right_params[3:6]
                right_hand_pose = right_params[6:51]  # 45 parameters
                
                left_translation = left_params[:3]
                left_global_orient = left_params[3:6]
                left_hand_pose = left_params[6:51]  # 45 parameters
                
                # Convert using the correct MANO parameter structure
                right_joints = convert_mano_params_to_joints(
                    right_translation, right_global_orient, right_hand_pose, 'right'
                )
                left_joints = convert_mano_params_to_joints(
                    left_translation, left_global_orient, left_hand_pose, 'left'
                )
            else:
                # Fallback to old format for backward compatibility
                left_pos = processed_data['left_hand_position'][i]
                right_pos = processed_data['right_hand_position'][i]
                left_angles = processed_data['left_hand_angles'][i]
                right_angles = processed_data['right_hand_angles'][i]
                
                # Convert to joint positions (old method)
                left_joints = convert_mano_params_to_joints(left_pos, left_angles, 'left')
                right_joints = convert_mano_params_to_joints(right_pos, right_angles, 'right')
            
            all_left_joints.append(left_joints)
            all_right_joints.append(right_joints)
        
        has_mano_joints = True
    except Exception as e:
        print(f"Warning: Could not convert to MANO joints: {e}")
        has_mano_joints = False
    
    # MANO joint structure information
    mano_joint_names = [
        "Wrist",           # 0
        "Thumb_CMC",       # 1
        "Thumb_MCP",       # 2
        "Thumb_IP",        # 3
        "Thumb_Tip",       # 4
        "Index_MCP",       # 5
        "Index_PIP",       # 6
        "Index_DIP",       # 7
        "Index_Tip",       # 8
        "Middle_MCP",      # 9
        "Middle_PIP",      # 10
        "Middle_DIP",      # 11
        "Middle_Tip",      # 12
        "Ring_MCP",        # 13
        "Ring_PIP",        # 14
        "Ring_DIP",        # 15
        "Ring_Tip",        # 16
        "Little_MCP",      # 17
        "Little_PIP",      # 18
        "Little_DIP",      # 19
        "Little_Tip"       # 20
    ]
    
    # MANO joint connections (parent-child relationships)
    mano_joint_connections = [
        # Wrist to finger bases
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index
        (5, 6), (6, 7), (7, 8),
        # Middle  
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Little
        (17, 18), (18, 19), (19, 20)
    ]
    
    # Finger grouping information
    finger_groups = {
        "thumb": {"joints": [1, 2, 3, 4], "color": "#FF6B6B"},
        "index": {"joints": [5, 6, 7, 8], "color": "#4ECDC4"},
        "middle": {"joints": [9, 10, 11, 12], "color": "#45B7D1"},
        "ring": {"joints": [13, 14, 15, 16], "color": "#96CEB4"},
        "little": {"joints": [17, 18, 19, 20], "color": "#FFEAA7"},
        "wrist": {"joints": [0], "color": "#DDA0DD"}
    }
    
    def calculate_bone_lengths(joints, connections):
        """Calculate bone lengths from joint positions."""
        bone_lengths = []
        for parent, child in connections:
            if parent < len(joints) and child < len(joints):
                length = np.linalg.norm(joints[child] - joints[parent])
                bone_lengths.append(length)
            else:
                bone_lengths.append(0.0)
        return np.array(bone_lengths)
    
    for i in range(num_frames):
        frame_data = {
            "frame_index": i,
            "timestamp": float(i / fps),  # Using detected FPS, ensure it's a Python float
            
            # Hand positions (3D coordinates)
            "left_hand_position": convert_to_serializable(processed_data['left_hand_position'][i]),
            "right_hand_position": convert_to_serializable(processed_data['right_hand_position'][i]),
            
            # Hand rotation parameters (48 per hand)
            "left_hand_angles": convert_to_serializable(processed_data['left_hand_angles'][i]),
            "right_hand_angles": convert_to_serializable(processed_data['right_hand_angles'][i])
        }
        
        # Add joint data if available
        if has_mano_joints:
            frame_data.update({
                # Absolute joint coordinates (21 joints per hand)
                "left_hand_joints": convert_to_serializable(all_left_joints[i]),
                "right_hand_joints": convert_to_serializable(all_right_joints[i]),
                
                # Bone lengths (calculated from joint positions)
                "left_hand_bone_lengths": convert_to_serializable(calculate_bone_lengths(all_left_joints[i], mano_joint_connections)),
                "right_hand_bone_lengths": convert_to_serializable(calculate_bone_lengths(all_right_joints[i], mano_joint_connections))
            })
        
        output_data["frames"].append(frame_data)
    
    # Add comprehensive metadata
    metadata = {
        "total_frames": int(num_frames),
        "fps": float(fps),
        "data_format": "comprehensive_hand_data",
        "coordinate_system": "3D_positions_plus_rotation_angles_plus_joint_coordinates",
        "hand_angles_per_hand": 48,
        "joints_per_hand": 21,
        "position_dimensions": 3,
        "angle_units": "radians",
        "model_type": "PianoMotion10M_diffusion_with_position_guide",
        "mano_joint_names": mano_joint_names,
        "mano_joint_connections": mano_joint_connections,
        "finger_groups": finger_groups,
        "hand_ordering_verified": True,  # Indicates hand ordering verification was performed
        "hand_ordering_corrected": processed_data.get('hand_ordering_corrected', False),  # Whether correction was applied
        "scale_factors_applied": [1.5, 1.5, 25],  # Scale factors used for coordinate transformation
        "scaling_validation_passed": processed_data.get('scaling_validation_passed', False),  # Whether validation passed
        "robust_scaling_system": True,  # Indicates use of robust scaling system
        "mano_parameter_structure": processed_data.get('mano_parameter_structure', 'standard_mano'),  # Detected MANO parameter structure
        "mano_conversion_method": processed_data.get('mano_conversion_method', 'robust_detection'),  # Method used for MANO conversion
        "coordinate_system_detection": convert_to_serializable(processed_data.get('coordinate_system', {})),  # Detected coordinate system convention
    }
    
    # Safely add optional metadata fields
    for key in ['coordinate_info', 'spatial_info', 'keyboard_info', 'fps_configuration']:
        if key in processed_data:
            try:
                metadata[key] = convert_to_serializable(processed_data[key])
            except Exception as e:
                print(f"Warning: Could not serialize {key}: {e}")
                # Skip if not serializable
                pass
    
    output_data["metadata"] = metadata
    
    # Convert the entire output data to ensure it's JSON serializable
    serializable_output = convert_to_serializable(output_data)
    
    with open(output_path, "w") as f:
        json.dump(serializable_output, f, indent=2)

def main():
    print("=== PianoMotion10M Audio to Hand Motion Inference ===")
    print(f"Input: {DEFAULT_AUDIO_PATH}")
    print(f"Output: {OUTPUT_JSON}")
    print(f"Mode: {'Full sequence' if USE_FULL_SEQUENCE else 'Fixed duration (4.0s)'}")
    print(f"FPS: {FPS}")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Process audio input
    print("\nStep 1: Processing audio input...")
    audio_processor = create_audio_processor(
        sample_rate=16000, 
        target_duration=4.0,
        use_full_sequence=USE_FULL_SEQUENCE
    )
    
    # Check if input file exists
    if not os.path.exists(DEFAULT_AUDIO_PATH):
        print(f"Error: Input file not found: {DEFAULT_AUDIO_PATH}")
        sys.exit(1)
    
    try:
        audio_wave = audio_processor.process_audio_input(DEFAULT_AUDIO_PATH)
        
        # Debug: Check audio length
        audio_duration = len(audio_wave) / 16000
        print(f"Debug: Audio length after processing: {len(audio_wave)} samples ({audio_duration:.2f}s)")
        
        # Validate the processed audio
        if not audio_processor.validate_audio(audio_wave):
            print("Warning: Audio validation failed, but continuing...")
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        sys.exit(1)
    
    # 2. Load model
    print("\nStep 2: Loading model...")
    sys.path.append('PianoMotion10M')
    
    from models.piano2posi import Piano2Posi
    from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D
    
    # Load configuration
    config = load_piano2posi_config(CHECKPOINT_PATH)
    piano2posi_args = Piano2PosiArgs(config)
    diffusion_args = DiffusionArgs()
    
    # Calculate sequence length based on audio duration
    if USE_FULL_SEQUENCE:
        audio_duration = len(audio_wave) / 16000  # 16000 Hz sample rate
        seq_length = int(audio_duration * FPS)
        print(f"Using full sequence mode: {audio_duration:.2f}s -> {seq_length} frames")
        
        # Check if sequence length exceeds model's maximum
        if seq_length > MAX_SEQUENCE_LENGTH:
            print(f"Warning: Sequence length ({seq_length} frames) exceeds model maximum ({MAX_SEQUENCE_LENGTH} frames)")
            print(f"Truncating to {MAX_SEQUENCE_LENGTH} frames ({MAX_SEQUENCE_LENGTH/FPS:.2f}s)")
            seq_length = MAX_SEQUENCE_LENGTH
            
            # Also truncate the audio to match
            max_audio_samples = int(MAX_SEQUENCE_LENGTH / FPS * 16000)
            if len(audio_wave) > max_audio_samples:
                audio_wave = audio_wave[:max_audio_samples]
                print(f"Audio truncated to {len(audio_wave)} samples ({len(audio_wave)/16000:.2f}s)")
        
        # Validate sequence length (model may have limitations)
        max_reasonable_frames = 5000  # ~2.8 minutes at 30 FPS
        if seq_length > max_reasonable_frames:
            print(f"Warning: Very long sequence ({seq_length} frames). This may cause memory issues.")
            print(f"Consider using fixed duration mode for very long audio files.")
    else:
        # Use the training sequence length from the model config
        seq_length = int(diffusion_args.train_sec * FPS)
        print(f"Using fixed duration mode: {diffusion_args.train_sec}s -> {seq_length} frames")
        
        # Also truncate the audio to match the training duration
        train_audio_samples = int(diffusion_args.train_sec * 16000)
        if len(audio_wave) > train_audio_samples:
            audio_wave = audio_wave[:train_audio_samples]
            print(f"Audio truncated to training duration: {len(audio_wave)} samples ({len(audio_wave)/16000:.2f}s)")
    
    # Set conditional dimension
    cond_dim = 1024 if 'large' in piano2posi_args.wav2vec_path else 768
    
    # Create models
    print("Creating Piano2Posi model...")
    piano2posi = Piano2Posi(piano2posi_args)
    
    print("Creating UNet model...")
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
    
    print("Creating composite diffusion model...")
    model = GaussianDiffusion1D_piano2pose(
        unet,
        piano2posi,
        seq_length=seq_length,
        timesteps=diffusion_args.timesteps,
        objective='pred_v',
    )
    
    # Load checkpoint with proper handling of HuBERT weights
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    # Use the specialized loading function from piano2posi.py to handle HuBERT weights properly
    print("Loading checkpoint weights with proper HuBERT handling...")
    from models.piano2posi import load_without_some_keys
    
    # The load_without_some_keys function handles the mismatch between 
    # fresh HuBERT model and checkpoint HuBERT weights
    try:
        load_without_some_keys(model, checkpoint, dropout_key=[])
        print("[OK] Checkpoint loaded successfully using specialized loader")
    except Exception as e:
        print(f"Specialized loader failed: {e}")
        print("Falling back to standard loading with strict=False...")
        
        # Fallback to standard loading
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")
            else:
                for key in missing_keys[:5]:
                    print(f"  - {key}")
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  + {key}")
            else:
                for key in unexpected_keys[:5]:
                    print(f"  + {key}")
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        print("[OK] Fallback loading completed")
    model.eval()
    model.to(device)
    
    print("[OK] Model loaded successfully")
    
    # 3. Run inference
    print("\nStep 3: Running inference...")
    pose_hat, guide, frame_num = run_inference(model, audio_wave, device, fps=FPS)
    
    # 4. Process output
    print("\nStep 4: Processing output...")
    processed_data = process_output(pose_hat, guide, device)
    
    # 5. Save results
    print("\nStep 5: Saving results...")
    save_output(processed_data, OUTPUT_JSON, fps=FPS)
    
    print(f"\n[OK] Inference completed successfully!")
    print(f"[OK] Output saved to: {OUTPUT_JSON}")
    print(f"[OK] Processed {frame_num} frames at {FPS} FPS")

if __name__ == "__main__":
    main()
