import os
import json
import numpy as np
import pretty_midi
import soundfile as sf
import torch
import copy
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_audio_pipeline(audio_wave, target_duration=4.0, sample_rate=16000):
    """
    Comprehensive audio preprocessing pipeline matching PianoPose dataset.
    """
    
    # Ensure correct sample rate
    if len(audio_wave) == 0:
        raise ValueError("Empty audio array")
    
    # Calculate target length in samples
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio_wave)
    
    print(f"Audio preprocessing:")
    print(f"  - Input length: {current_samples} samples ({current_samples/sample_rate:.2f}s)")
    print(f"  - Target length: {target_samples} samples ({target_duration:.2f}s)")
    
    # Handle duration mismatch
    if current_samples > target_samples:
        # For longer audio, take the first segment (could be improved with onset detection)
        audio_wave = audio_wave[:target_samples]
        print(f"  - Trimmed to {target_duration}s")
    elif current_samples < target_samples:
        # For shorter audio, pad with silence
        padding = target_samples - current_samples
        audio_wave = np.pad(audio_wave, (0, padding), mode='constant', constant_values=0)
        print(f"  - Padded with {padding} samples of silence")
    
    # Normalize audio to [-1, 1] range (critical for model)
    max_amplitude = np.max(np.abs(audio_wave))
    if max_amplitude > 0:
        audio_wave = audio_wave / max_amplitude
        print(f"  - Normalized by factor {max_amplitude:.4f}")
    else:
        print("  - Warning: Silent audio detected")
    
    # Apply additional preprocessing to match training data
    # Remove DC offset
    audio_wave = audio_wave - np.mean(audio_wave)
    
    # Apply gentle high-pass filter to remove low-frequency noise
    try:
        from scipy.signal import butter, filtfilt
        # High-pass filter at 80 Hz (below piano range)
        nyquist = sample_rate / 2
        high_cutoff = 80.0 / nyquist
        b, a = butter(2, high_cutoff, btype='high')
        audio_wave = filtfilt(b, a, audio_wave)
        print(f"  - Applied high-pass filter at 80 Hz")
        
        # Re-normalize after filtering to ensure [-1, 1] range
        max_amplitude = np.max(np.abs(audio_wave))
        if max_amplitude > 0:
            audio_wave = audio_wave / max_amplitude
            print(f"  - Re-normalized after filtering by factor {max_amplitude:.4f}")
    except Exception as e:
        print(f"  - Warning: Could not apply filter: {e}")
    
    # Final validation
    assert len(audio_wave) == target_samples, f"Audio length mismatch: {len(audio_wave)} != {target_samples}"
    assert np.max(np.abs(audio_wave)) <= 1.0, f"Audio not properly normalized: max = {np.max(np.abs(audio_wave))}"
    
    print(f"  ✓ Audio preprocessing completed")
    return audio_wave.astype(np.float32)

def calculate_frame_number(audio_length, sample_rate=16000, fps=30, train_sec=4):
    """Calculate the number of frames for inference."""
    
    # Calculate actual audio duration
    audio_duration = audio_length / sample_rate
    
    # Use minimum of actual duration or training duration
    effective_duration = min(audio_duration, train_sec)
    
    # Calculate frame count
    frame_num = int(effective_duration * fps)
    
    print(f"Audio duration: {audio_duration:.2f}s, Effective: {effective_duration:.2f}s, Frames: {frame_num}")
    
    return frame_num

def validate_sample_method(model, audio_tensor, frame_num, batch_size):
    """Validate that the model sample method works with given parameters."""
    
    try:
        # Test the sample method signature with proper audio format
        # Piano2Posi expects audio as (batch, samples) - 1D audio per batch
        with torch.no_grad():
            test_pose, test_guide = model.sample(audio_tensor, frame_num, batch_size)
        
        print(f"✓ Sample method validation passed:")
        print(f"  - Input audio shape: {audio_tensor.shape}")
        print(f"  - Requested frames: {frame_num}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Output pose shape: {test_pose.shape}")
        print(f"  - Output guide shape: {test_guide.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample method validation failed: {e}")
        return False

def run_model_inference(model, audio_wave, device, train_sec=4):
    """Run model inference with correct method signature."""
    
    try:
        # Apply comprehensive audio preprocessing
        audio_wave = preprocess_audio_pipeline(audio_wave, target_duration=train_sec)
        
        # Convert to tensor with proper dimensions - Piano2Posi expects 1D audio
        audio_tensor = torch.tensor(audio_wave, dtype=torch.float32).unsqueeze(0)  # (1, samples)
        audio_tensor = audio_tensor.to(device)
        
        # Calculate proper frame number based on audio duration
        frame_num = calculate_frame_number(len(audio_wave), train_sec=train_sec)
        
        # Validate inputs
        print(f"Input validation:")
        print(f"  - Audio tensor shape: {audio_tensor.shape}")
        print(f"  - Frame number: {frame_num}")
        print(f"  - Device: {device}")
        
        with torch.no_grad():
            # Use correct sample method signature - batch_size should be 1 for single inference
            batch_size = 1
            pose_hat, guide = model.sample(audio_tensor, frame_num, batch_size)
            
            # Ensure correct output dimensions - model returns (batch, channels, frames)
            # We need to permute to (batch, frames, channels) for processing
            if pose_hat.dim() == 3:
                pose_hat = pose_hat.permute(0, 2, 1)  # (batch, frames, 96)
            if guide.dim() == 3:
                guide = guide.permute(0, 2, 1)        # (batch, frames, 6)
        
        print(f"✓ Inference completed successfully")
        print(f"  - Output pose shape: {pose_hat.shape}")
        print(f"  - Output guide shape: {guide.shape}")
        
        return pose_hat, guide, frame_num
        
    except Exception as e:
        print(f"✗ Error during model inference: {e}")
        print(f"Audio tensor shape: {audio_tensor.shape if 'audio_tensor' in locals() else 'Not created'}")
        print(f"Frame number: {frame_num if 'frame_num' in locals() else 'Not calculated'}")
        raise

def ensure_audio_frame_sync(audio_wave, frame_num, sample_rate=16000, fps=30):
    """Ensure audio length matches expected frame count."""
    
    expected_audio_length = int((frame_num / fps) * sample_rate)
    current_audio_length = len(audio_wave)
    
    if current_audio_length != expected_audio_length:
        print(f"Adjusting audio length: {current_audio_length} → {expected_audio_length} samples")
        
        if current_audio_length > expected_audio_length:
            # Trim audio
            audio_wave = audio_wave[:expected_audio_length]
        else:
            # Pad audio
            padding = expected_audio_length - current_audio_length
            audio_wave = np.pad(audio_wave, (0, padding), mode='constant')
    
    return audio_wave

def synchronize_audio_with_frames(audio_wave, midi_path, fps=30, sample_rate=16000):
    """
    Synchronize audio timing with frame indices for proper model input.
    """
    
    try:
        # Load MIDI to get timing information
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Calculate timing parameters
        audio_duration = len(audio_wave) / sample_rate
        midi_duration = pm.get_end_time()
        frame_count = int(audio_duration * fps)
        
        print(f"Synchronization info:")
        print(f"  - Audio duration: {audio_duration:.2f}s")
        print(f"  - MIDI duration: {midi_duration:.2f}s")
        print(f"  - Frame count: {frame_count} frames")
        print(f"  - FPS: {fps}")
        
        # Check for timing mismatches
        if abs(audio_duration - midi_duration) > 0.5:  # 500ms tolerance
            print(f"  - Warning: Audio-MIDI duration mismatch: {abs(audio_duration - midi_duration):.2f}s")
        
        # Create frame timing array
        frame_times = np.linspace(0, audio_duration, frame_count)
        
        # Validate synchronization
        samples_per_frame = sample_rate / fps
        expected_samples = frame_count * samples_per_frame
        actual_samples = len(audio_wave)
        
        if abs(expected_samples - actual_samples) > sample_rate * 0.1:  # 100ms tolerance
            print(f"  - Warning: Frame-audio sync issue: {abs(expected_samples - actual_samples)/sample_rate:.2f}s difference")
        
        return frame_times, frame_count
        
    except Exception as e:
        print(f"Synchronization error: {e}")
        # Fallback to simple calculation
        audio_duration = len(audio_wave) / sample_rate
        frame_count = int(audio_duration * fps)
        frame_times = np.linspace(0, audio_duration, frame_count)
        return frame_times, frame_count

def synthesize_piano_from_midi(midi_path, sample_rate=16000, soundfont_path=None):
    """
    Improved MIDI to audio synthesis with realistic piano sounds.
    """
    
    try:
        # Load MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        print(f"MIDI loaded: {len(pm.instruments)} instruments, duration: {pm.get_end_time():.2f}s")
        
        # Option 1: Use fluidsynth with a piano soundfont (if available)
        if soundfont_path and os.path.exists(soundfont_path):
            try:
                audio_wave = pm.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
                print("✓ Used FluidSynth with soundfont for realistic piano sound")
                return audio_wave
            except Exception as e:
                print(f"FluidSynth failed: {e}, falling back to synthesize")
        
        # Option 2: Use pretty_midi's built-in synthesis with better waveform
        # Try multiple waveforms and pick the best one
        waveforms = [
            ('sine', np.sin),
            ('triangle', lambda x: 2 * np.arcsin(np.sin(x)) / np.pi),
            ('sawtooth', lambda x: 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5)))
        ]
        
        best_audio = None
        best_name = None
        
        for name, waveform in waveforms:
            try:
                audio_wave = pm.synthesize(fs=sample_rate, wave=waveform)
                # Simple quality check: prefer non-zero audio with reasonable amplitude
                if len(audio_wave) > 0 and np.max(np.abs(audio_wave)) > 0.001:
                    best_audio = audio_wave
                    best_name = name
                    break
            except Exception as e:
                print(f"Synthesis with {name} wave failed: {e}")
                continue
        
        if best_audio is not None:
            print(f"✓ Used {best_name} wave synthesis")
            return best_audio.astype(np.float32)
        else:
            # Last resort: silent audio
            duration = max(pm.get_end_time(), 4.0)  # At least 4 seconds
            print("Warning: All synthesis methods failed, generating silence")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)
            
    except Exception as e:
        print(f"MIDI synthesis error: {e}")
        # Generate silence as fallback
        return np.zeros(int(4.0 * sample_rate), dtype=np.float32)

def validate_audio_quality(audio_wave, sample_rate=16000):
    """
    Validate audio quality and characteristics.
    """
    
    # Basic quality checks
    duration = len(audio_wave) / sample_rate
    max_amplitude = np.max(np.abs(audio_wave))
    rms = np.sqrt(np.mean(audio_wave**2))
    
    print(f"Audio quality validation:")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Max amplitude: {max_amplitude:.4f}")
    print(f"  - RMS level: {rms:.4f}")
    print(f"  - Dynamic range: {20*np.log10(max_amplitude/rms):.1f} dB" if rms > 0 else "  - Dynamic range: N/A (silent)")
    
    # Quality warnings
    warnings = []
    
    if max_amplitude < 0.01:
        warnings.append("Very low amplitude - audio might be too quiet")
    elif max_amplitude > 0.99:
        warnings.append("Near-clipping amplitude - audio might be too loud")
    
    if rms < 0.001:
        warnings.append("Very low RMS - audio might be mostly silent")
    
    if duration < 3.0:
        warnings.append("Short duration - might not provide enough context")
    
    # Check for silence
    silence_threshold = 0.0001
    silent_samples = np.sum(np.abs(audio_wave) < silence_threshold)
    silence_ratio = silent_samples / len(audio_wave)
    
    if silence_ratio > 0.5:
        warnings.append(f"High silence ratio: {silence_ratio:.1%}")
    
    # Frequency content check (basic spectral analysis)
    try:
        freqs = np.fft.fftfreq(len(audio_wave), 1/sample_rate)
        fft = np.abs(np.fft.fft(audio_wave))
        
        # Check for reasonable frequency distribution
        low_freq_energy = np.sum(fft[(freqs >= 80) & (freqs <= 500)])
        mid_freq_energy = np.sum(fft[(freqs >= 500) & (freqs <= 2000)])
        high_freq_energy = np.sum(fft[(freqs >= 2000) & (freqs <= 8000)])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            print(f"  - Frequency distribution: Low: {low_freq_energy/total_energy:.1%}, Mid: {mid_freq_energy/total_energy:.1%}, High: {high_freq_energy/total_energy:.1%}")
        
    except Exception as e:
        warnings.append(f"Could not analyze frequency content: {e}")
    
    # Report warnings
    if warnings:
        print("  ⚠ Warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  ✓ Audio quality looks good")
    
    return len(warnings) == 0

# Paths (adjust these to your setup)
MIDI_PATH = "examples\\example_2.mid"            # Input MIDI file
AUDIO_PATH = "examples\\example_2.wav"           # Temporary audio file path
CHECKPOINT_PATH = "PianoMotion10M\\checkpoints\\diffusion_posiguide_hubertlarge_tf2\\piano2pose-iter=90000-val_loss=0.0364401508122683.ckpt"  # Downloaded model checkpoint
OUTPUT_JSON = "predicted_hand_motion.json"      # Output JSON path

def validate_model_config(piano2posi_args, diffusion_args, checkpoint_path):
    """Validate that model configuration matches checkpoint."""
    
    # Check if model type in checkpoint matches configuration
    if 'large' in checkpoint_path.lower():
        assert piano2posi_args.num_layer == 8, "Large model should have 8 layers"
        assert diffusion_args.unet_dim == 256, "Large model should have unet_dim=256"
    elif 'base' in checkpoint_path.lower():
        assert piano2posi_args.num_layer == 4, "Base model should have 4 layers"
        assert diffusion_args.unet_dim == 128, "Base model should have unet_dim=128"
    
    # Validate critical dimensions
    assert piano2posi_args.bs_dim == 6, "Piano2Posi must use bs_dim=6"
    assert diffusion_args.bs_dim == 96, "Diffusion model must use bs_dim=96"
    
    print("✓ Model configuration validation passed")

class Piano2PosiArgs:
    def __init__(self, model_type='large'):
        self.feature_dim = 512
        self.bs_dim = 6  # CRITICAL: Position predictor always uses 6 (xyz for both hands)
        self.loss_mode = 'naive_l2'
        self.encoder_type = 'transformer'
        self.hidden_type = 'audio_f'
        self.max_seq_len = 1500
        self.period = 1500
        self.latest_layer = "tanh"
        
        # Configure based on model type
        if model_type == 'large':
            self.num_layer = 8
            self.wav2vec_path = "facebook/hubert-large-ls960-ft"
        elif model_type == 'base':
            self.num_layer = 4
            self.wav2vec_path = "facebook/hubert-base-ls960"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class DiffusionArgs:
    def __init__(self, model_type='large'):
        self.timesteps = 1000
        self.train_sec = 4
        self.xyz_guide = True
        self.remap_noise = True
        self.bs_dim = 96  # Gesture generator always uses 96 (hand rotation parameters)
        self.encoder_type = 'transformer'
        self.experiment_name = "midi_inference"
        
        # Configure based on model type
        if model_type == 'large':
            self.unet_dim = 256
            self.num_layer = 8
        elif model_type == 'base':
            self.unet_dim = 128  # Smaller for base model
            self.num_layer = 4
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def validate_model_output(pose_hat, guide):
    """Validate model output dimensions and format."""
    
    # Check dimensions - after transpose, shape is (batch, frames, channels)
    assert pose_hat.shape[2] == 96, f"Expected 96 rotation parameters, got {pose_hat.shape[2]}"
    assert guide.shape[2] == 6, f"Expected 6 position coordinates, got {guide.shape[2]}"
    assert pose_hat.shape[0] == guide.shape[0], "Batch size mismatch between pose and guide"
    assert pose_hat.shape[1] == guide.shape[1], "Frame count mismatch between pose and guide"
    
    print(f"✓ Model output validation passed:")
    print(f"  - Pose shape: {pose_hat.shape} (batch, frames, 96_angles)")
    print(f"  - Guide shape: {guide.shape} (batch, frames, 6_positions)")
    print(f"  - Frame count: {pose_hat.shape[1]}")

def apply_coordinate_transformations(pose_hat, guide, device='cpu'):
    """
    Apply proper scaling and coordinate transformations to model outputs.
    
    Based on official PianoMotion10M inference code.
    """
    
    # Official scaling factors from PianoMotion10M
    scale = torch.tensor([1.5, 1.5, 25]).to(device)
    
    print(f"Coordinate transformation:")
    print(f"  - Scale factors: {scale.tolist()}")
    print(f"  - Input pose shape: {pose_hat.shape}")
    print(f"  - Input guide shape: {guide.shape}")
    
    # Convert pose predictions to radians (rotation angles)
    # pose_hat has shape (batch, frames, pose_params), no need to transpose
    prediction = pose_hat[0].detach().cpu().numpy() * np.pi
    print(f"  - Converted pose to radians (range: [{np.min(prediction):.3f}, {np.max(prediction):.3f}])")
    
    # Apply scaling to guide positions (hand positions in world space)
    # scale.repeat(2) creates [1.5, 1.5, 25, 1.5, 1.5, 25] for both hands
    # Guide has shape (batch, frames, 6), so we need to expand scale to (1, frames, 6)
    scale_expanded = scale.repeat(2).view(1, 1, 6).expand_as(guide)  # Expand to match guide shape
    scaled_guide = (guide * scale_expanded)[0].cpu().numpy()  # Shape: (frames, 6)
    print(f"  - Applied scaling to guide positions")
    print(f"  - Guide range before scaling: [{np.min(guide[0].cpu().numpy()):.3f}, {np.max(guide[0].cpu().numpy()):.3f}]")
    print(f"  - Guide range after scaling: [{np.min(scaled_guide):.3f}, {np.max(scaled_guide):.3f}]")
    
    return prediction, scaled_guide

def understand_coordinate_system():
    """
    Explain the coordinate system used by PianoMotion10M.
    """
    
    coord_info = {
        "world_space": {
            "origin": "Piano keyboard center or camera position",
            "x_axis": "Left-right relative to piano keyboard",
            "y_axis": "Up-down (vertical)",
            "z_axis": "Forward-backward (depth from camera/piano)",
            "units": "Scaled units (not meters)",
            "scale_factors": [1.5, 1.5, 25]
        },
        "hand_positions": {
            "format": "3D coordinates (x, y, z) for each hand",
            "reference": "World space coordinates",
            "scaling": "Applied via scale.repeat(2) for both hands"
        },
        "hand_rotations": {
            "format": "48 parameters per hand (MANO-style)",
            "units": "Radians (multiplied by π)",
            "parameterization": "Rotation angles for hand joints"
        }
    }
    
    print("Coordinate System Information:")
    for category, details in coord_info.items():
        print(f"  {category.replace('_', ' ').title()}:")
        for key, value in details.items():
            print(f"    - {key.replace('_', ' ').title()}: {value}")
    
    return coord_info

def interpret_spatial_scale(scaled_positions, scale_factors=[1.5, 1.5, 25]):
    """
    Interpret the spatial scale of hand positions relative to piano playing.
    """
    
    # Extract positions for both hands
    right_pos = scaled_positions[:, :3]  # First 3 dimensions
    left_pos = scaled_positions[:, 3:]   # Last 3 dimensions
    
    # Calculate spatial statistics
    right_range = np.ptp(right_pos, axis=0)  # Range in each dimension
    left_range = np.ptp(left_pos, axis=0)
    
    # Estimate physical dimensions (approximate conversion)
    # These are rough estimates based on typical piano playing
    estimated_scale = {
        'x': 0.01,  # ~1cm per unit (hand lateral movement)
        'y': 0.01,  # ~1cm per unit (hand vertical movement)  
        'z': 0.004  # ~4mm per unit (hand depth movement, scaled by 25)
    }
    
    print(f"Spatial Scale Analysis:")
    print(f"  Scale factors applied: {scale_factors}")
    print(f"  Right hand range: X={right_range[0]:.2f}, Y={right_range[1]:.2f}, Z={right_range[2]:.2f}")
    print(f"  Left hand range: X={left_range[0]:.2f}, Y={left_range[1]:.2f}, Z={left_range[2]:.2f}")
    
    # Convert to estimated physical dimensions
    estimated_scale_array = np.array([estimated_scale['x'], estimated_scale['y'], estimated_scale['z']])
    right_physical = right_range * estimated_scale_array
    left_physical = left_range * estimated_scale_array
    
    print(f"  Estimated physical range (meters):")
    print(f"    Right hand: X={right_physical[0]:.3f}m, Y={right_physical[1]:.3f}m, Z={right_physical[2]:.3f}m")
    print(f"    Left hand: X={left_physical[0]:.3f}m, Y={left_physical[1]:.3f}m, Z={left_physical[2]:.3f}m")
    
    return {
        'right_range': right_range,
        'left_range': left_range,
        'right_physical': right_physical,
        'left_physical': left_physical,
        'scale_factors': scale_factors
    }

def add_piano_keyboard_context(hand_positions, keyboard_width=1.2, key_count=88):
    """
    Add spatial context relative to piano keyboard for better understanding.
    """
    
    # Standard piano keyboard dimensions (approximate)
    white_key_width = 0.023  # ~23mm
    white_key_length = 0.150  # ~150mm
    black_key_width = 0.013  # ~13mm
    black_key_length = 0.095  # ~95mm
    
    keyboard_info = {
        'total_width': keyboard_width,
        'key_count': key_count,
        'white_keys': 52,
        'white_key_width': white_key_width,
        'approximate_octave_width': white_key_width * 7  # C to B
    }
    
    print(f"Piano Keyboard Spatial Context:")
    print(f"  Total keyboard width: {keyboard_width:.2f}m")
    print(f"  Octave width: ~{keyboard_info['approximate_octave_width']:.3f}m")
    print(f"  White key width: {white_key_width:.3f}m")
    
    # Analyze hand position relative to keyboard
    right_pos = hand_positions[:, :3]
    left_pos = hand_positions[:, 3:]
    
    # Calculate hand separation
    hand_separation = np.linalg.norm(np.mean(right_pos, axis=0) - np.mean(left_pos, axis=0))
    
    # Estimate keyboard coverage
    total_hand_span = np.ptp(np.concatenate([right_pos[:, 0], left_pos[:, 0]]))  # X-dimension span
    
    print(f"  Hand separation: {hand_separation:.2f} units")
    print(f"  Total hand span: {total_hand_span:.2f} units")
    
    # Rough estimate of keyboard coverage
    estimated_scale = {'x': 0.01}  # ~1cm per unit
    estimated_octaves = total_hand_span / (keyboard_info['approximate_octave_width'] / estimated_scale['x'])
    print(f"  Estimated keyboard coverage: ~{estimated_octaves:.1f} octaves")
    
    return keyboard_info

def process_model_output_with_proper_coordinates(pose_hat, guide, device='cpu'):
    """
    Process model output with proper coordinate system handling.
    """
    
    # Display coordinate system information
    coord_info = understand_coordinate_system()
    
    # Apply proper transformations
    prediction, scaled_guide = apply_coordinate_transformations(pose_hat, guide, device)
    
    # Apply smoothing (from official code)
    from scipy.signal import savgol_filter
    for i in range(prediction.shape[1]):
        prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
    for i in range(scaled_guide.shape[1]):
        scaled_guide[:, i] = savgol_filter(scaled_guide[:, i], 5, 2)
    
    # Split data correctly
    right_hand_angles = prediction[:, :48]  # 48 rotation parameters
    left_hand_angles = prediction[:, 48:]   # 48 rotation parameters
    right_hand_pos = scaled_guide[:, :3]    # Scaled xyz position
    left_hand_pos = scaled_guide[:, 3:]     # Scaled xyz position
    
    # Analyze spatial scale
    spatial_info = interpret_spatial_scale(scaled_guide)
    
    # Add piano context
    keyboard_info = add_piano_keyboard_context(scaled_guide)
    
    return {
        'right_hand_angles': right_hand_angles,
        'left_hand_angles': left_hand_angles,
        'right_hand_position': right_hand_pos,
        'left_hand_position': left_hand_pos,
        'num_frames': prediction.shape[0],
        'coordinate_info': coord_info,
        'spatial_info': spatial_info,
        'keyboard_info': keyboard_info
    }

def process_model_output(pose_hat, guide):
    """Process model output according to official format with proper coordinate handling."""
    
    # Use the comprehensive coordinate system handling
    return process_model_output_with_proper_coordinates(pose_hat, guide, device='cpu')

def save_output_to_json(processed_data, output_path):
    """Save processed data in comprehensive format for standalone visualization."""
    
    output_data = {"frames": []}
    num_frames = processed_data['num_frames']
    
    # Import MANO conversion to get joint data
    from mano_utils import convert_mano_params_to_joints
    
    # Get joint data for all frames
    print("Converting MANO parameters to joint positions for all frames...")
    all_left_joints = []
    all_right_joints = []
    
    for i in range(num_frames):
        left_pos = processed_data['left_hand_position'][i]
        right_pos = processed_data['right_hand_position'][i]
        left_angles = processed_data['left_hand_angles'][i]
        right_angles = processed_data['right_hand_angles'][i]
        
        # Convert to joint positions
        left_joints = convert_mano_params_to_joints(left_pos, left_angles, 'left')
        right_joints = convert_mano_params_to_joints(right_pos, right_angles, 'right')
        
        all_left_joints.append(left_joints)
        all_right_joints.append(right_joints)
    
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
    
    for i in range(num_frames):
        frame_data = {
            "frame_index": i,
            "timestamp": i / 30.0,  # Assuming 30 FPS
            
            # Hand positions (3D coordinates)
            "left_hand_position": processed_data['left_hand_position'][i].tolist(),
            "right_hand_position": processed_data['right_hand_position'][i].tolist(),
            
            # Hand rotation parameters (48 per hand)
            "left_hand_angles": processed_data['left_hand_angles'][i].tolist(),
            "right_hand_angles": processed_data['right_hand_angles'][i].tolist(),
            
            # Absolute joint coordinates (21 joints per hand)
            "left_hand_joints": all_left_joints[i].tolist(),
            "right_hand_joints": all_right_joints[i].tolist(),
            
            # Bone lengths (calculated from joint positions)
            "left_hand_bone_lengths": calculate_bone_lengths(all_left_joints[i], mano_joint_connections).tolist(),
            "right_hand_bone_lengths": calculate_bone_lengths(all_right_joints[i], mano_joint_connections).tolist()
        }
        output_data["frames"].append(frame_data)
    
    # Add comprehensive metadata
    metadata = {
        "total_frames": num_frames,
        "fps": 30,
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
        "notes": [
            "Hand positions are 3D coordinates in world space",
            "Hand angles are MANO rotation parameters (48 per hand)",
            "Joint coordinates are absolute 3D positions (21 per hand)",
            "Bone lengths are calculated from joint positions",
            "Scaling factors [1.5, 1.5, 25] applied to positions",
            "All data needed for standalone visualization included"
        ]
    }
    
    # Safely add optional metadata fields
    for key in ['coordinate_info', 'spatial_info', 'keyboard_info']:
        if key in processed_data:
            try:
                # Convert numpy arrays to lists if present
                value = processed_data[key]
                if isinstance(value, dict):
                    converted_value = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            converted_value[k] = v.tolist()
                        else:
                            converted_value[k] = v
                    metadata[key] = converted_value
                else:
                    metadata[key] = value
            except:
                # Skip if not serializable
                pass
    
    output_data["metadata"] = metadata
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

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

def main():
    print("=== PianoMotion10M MIDI to Hand Motion Inference ===")
    
    # Determine model type from checkpoint
    model_type = 'large' if 'large' in CHECKPOINT_PATH else 'base'
    print(f"Using {model_type} model configuration")
    
    # 1. Convert MIDI to audio with improved synthesis
    print("Step 1: Converting MIDI to audio...")
    try:
        # Try to use realistic piano synthesis
        audio_wave = synthesize_piano_from_midi(MIDI_PATH, sample_rate=16000)
        
        # Validate initial audio
        if len(audio_wave) == 0:
            raise ValueError("Audio synthesis produced empty result")
        
        # Apply comprehensive preprocessing
        audio_wave = preprocess_audio_pipeline(audio_wave, target_duration=4.0)
        
        # Validate audio quality
        quality_ok = validate_audio_quality(audio_wave)
        if not quality_ok:
            print("⚠ Audio quality warnings detected, but proceeding...")
        
        # Synchronize with frame timing
        frame_times, expected_frames = synchronize_audio_with_frames(
            audio_wave, MIDI_PATH, fps=30
        )
        
        # Save processed audio for inspection
        sf.write(AUDIO_PATH, audio_wave, samplerate=16000)
        print(f"✓ Processed audio saved to: {AUDIO_PATH}")
        print(f"✓ Audio duration: {len(audio_wave)/16000:.2f}s, Expected frames: {expected_frames}")
        
    except Exception as e:
        print(f"✗ Error in audio processing: {e}")
        return

    # 2. Load PianoMotion10M model with proper architecture
    print("\nStep 2: Loading PianoMotion10M model checkpoint...")
    try:
        import sys
        sys.path.append('PianoMotion10M')  # Add PianoMotion10M to Python path

        from models.piano2posi import Piano2Posi
        from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D

        # Create proper configurations
        piano2posi_args = Piano2PosiArgs(model_type)
        diffusion_args = DiffusionArgs(model_type)
        
        # Validate configuration
        validate_model_config(piano2posi_args, diffusion_args, CHECKPOINT_PATH)
        
        # Set conditional dimension based on wav2vec model
        if 'large' in piano2posi_args.wav2vec_path:
            cond_dim = 1024
        elif 'base' in piano2posi_args.wav2vec_path:
            cond_dim = 768
        else:
            raise ValueError("Unknown wav2vec model type")

        # Create Piano2Posi model
        piano2posi = Piano2Posi(piano2posi_args)

        # Create Unet1D model
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
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
            
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()

        # Force CPU usage to avoid CUDA compatibility issues
        device = torch.device('cpu')
        model.to(device)
        print(f"✓ Model loaded on device: {device}")
        print(f"✓ Using transformer architecture with {diffusion_args.num_layer} layers")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # 3. Run inference
    print("\nStep 3: Running inference on audio...")
    try:
        # Validate sample method before running inference
        # Create test audio tensor with proper format (batch, samples)
        audio_test = torch.zeros(1, 64000).to(device)  # 4 seconds at 16kHz
        if not validate_sample_method(model, audio_test, 120, 1):
            print("✗ Model sample method validation failed")
            return
        
        # Run actual inference with proper method signature
        pose_hat, guide, actual_frame_num = run_model_inference(
            model, audio_wave, device, train_sec=diffusion_args.train_sec
        )
        
        # Validate output dimensions
        validate_model_output(pose_hat, guide)
        
        # Import MANO integration functions
        from mano_utils import complete_mano_integration_pipeline
        
        # Use complete MANO integration pipeline
        processed_data, integration_info = complete_mano_integration_pipeline(
            pose_hat, guide, audio_wave, device
        )
        
        print(f"✓ Model inference completed with MANO integration:")
        print(f"  - Processed {actual_frame_num} frames")
        print(f"  - Duration: {actual_frame_num / 30:.2f} seconds")
        print(f"  - Right hand: {processed_data['right_hand_angles'].shape[1]} MANO rotation parameters")
        print(f"  - Left hand: {processed_data['left_hand_angles'].shape[1]} MANO rotation parameters") 
        print(f"  - Total frames: {processed_data['num_frames']}")
        print(f"  - Using proper MANO forward kinematics for joint positions")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return

    # 4. Save output to JSON
    print("\nStep 4: Saving results...")
    try:
        save_output_to_json(processed_data, OUTPUT_JSON)
        print(f"✓ Saved motion data to {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return

    # 5. (Optional) Data Analysis and Information
    print("\nStep 5: Data Analysis...")
    try:
        # Print detailed information about the output format
        print("✓ Output Data Analysis with MANO Integration:")
        print(f"  - Data format: MANO parameterization (positions + rotation angles)")
        print(f"  - Right hand: {processed_data['right_hand_angles'].shape[1]} MANO rotation parameters")
        print(f"  - Left hand: {processed_data['left_hand_angles'].shape[1]} MANO rotation parameters")
        print(f"  - Position scaling applied: [1.5, 1.5, 25]")
        print(f"  - Angle units: radians")
        print(f"  - Total frames: {processed_data['num_frames']}")
        print(f"  - MANO forward kinematics: Applied for joint position calculation")
        
        # Show sample data from first frame
        print("\n✓ Sample data from first frame:")
        print(f"  - Right hand position: {processed_data['right_hand_position'][0][:3]}")
        print(f"  - Left hand position: {processed_data['left_hand_position'][0][:3]}")
        print(f"  - Right hand angles (first 5): {processed_data['right_hand_angles'][0][:5]}")
        print(f"  - Left hand angles (first 5): {processed_data['left_hand_angles'][0][:5]}")
        
        print("\n✓ MANO Integration Status:")
        print("   - Rotation parameters converted to 21-joint positions via MANO forward kinematics")
        print("   - Anatomically correct hand structure with proper bone connections")
        print("   - Fallback to approximate joints if full MANO model unavailable")
        print("   - Compatible with official PianoMotion10M rendering pipeline")
        
    except Exception as e:
        print(f"⚠ Warning: Could not analyze data: {e}")
        print("Results saved to JSON file successfully")

    print("\n" + "=" * 50)
    print("✓ Inference completed successfully!")
    print(f"✓ Output saved to: {OUTPUT_JSON}")
    print(f"✓ Joint mapping info saved to: joint_mapping_info.json")
    print("=" * 50)

if __name__ == "__main__":
    main()
