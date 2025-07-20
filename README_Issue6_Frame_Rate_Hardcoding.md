# Issue 6: Frame Rate Hardcoding

## Problem Description

The current implementation hardcodes the frame rate at 30 FPS without verification:

```python
# Current code in midi_to_frames.py and mano_utils.py
"timestamp": i / 30.0,  # Assuming 30 FPS
frame_count = int(audio_duration * fps)  # fps=30
fps=30  # Hardcoded in multiple locations
```

**Issue**: The 30 FPS assumption is used throughout the codebase but may not match the actual frame rate used during model training. This affects timing calculations, frame synchronization, and temporal analysis.

## Why This Needs Verification

1. **Temporal Misalignment**: Wrong FPS leads to incorrect timing between audio and motion
2. **Frame Count Errors**: Calculations of expected frame counts will be wrong
3. **Synchronization Issues**: Audio-visual synchronization depends on correct frame rate
4. **Animation Errors**: Playback speed will be incorrect for visualization
5. **Metadata Accuracy**: Timestamps and duration calculations will be wrong

## Current Usage of Frame Rate

The FPS value is used in multiple locations:

### 1. **Frame Count Calculation**
```python
def calculate_frame_number(audio_length, sample_rate=16000, fps=30, train_sec=4):
    audio_duration = audio_length / sample_rate
    effective_duration = min(audio_duration, train_sec)
    frame_num = int(effective_duration * fps)
    return frame_num
```

### 2. **Timestamp Generation**
```python
# In save_output_to_json
"timestamp": i / 30.0,  # Assumes 30 FPS

# In synchronization
frame_times = np.linspace(0, audio_duration, frame_count)
```

### 3. **Sequence Length Configuration**
```python
seq_length=diffusion_args.train_sec * 30,  # Assumes 30 FPS
```

### 4. **Audio-Frame Synchronization**
```python
def synchronize_audio_with_frames(audio_wave, midi_path, fps=30, sample_rate=16000):
    frame_count = int(audio_duration * fps)
    samples_per_frame = sample_rate / fps
```

## Verification Steps

### Step 1: Check Model Training Configuration
```python
def verify_model_training_fps():
    """Verify the FPS used during model training."""
    
    try:
        import sys
        sys.path.append('PianoMotion10M')
        
        # Check diffusion model configuration
        from models.denoise_diffusion import GaussianDiffusion1D_piano2pose
        
        # Look for FPS or sequence length hints
        print("Checking model training configuration:")
        
        # Method 1: Check if FPS is stored in model configuration
        # This would be in the model's hyperparameters or args
        
        # Method 2: Check training scripts for FPS configuration
        training_files = [
            'PianoMotion10M/train.py',
            'PianoMotion10M/train_diffusion.py',
            'PianoMotion10M/eval.py'
        ]
        
        for file_path in training_files:
            if os.path.exists(file_path):
                print(f"\nAnalyzing {file_path}:")
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Look for FPS mentions
                    fps_patterns = [
                        r'fps\s*=\s*(\d+)',
                        r'frame.*rate\s*=\s*(\d+)',
                        r'(\d+)\s*fps',
                        r'seq.*length.*(\d+)',
                        r'train.*sec\s*\*\s*(\d+)'
                    ]
                    
                    for pattern in fps_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            print(f"  Found FPS pattern '{pattern}': {matches}")
        
        return True
        
    except Exception as e:
        print(f"Could not verify model training FPS: {e}")
        return False

def check_dataset_fps_configuration():
    """Check the dataset implementation for FPS configuration."""
    
    try:
        import sys
        sys.path.append('PianoMotion10M')
        from datasets.PianoPose import PianoPose
        
        # Create dataset instance
        dataset = PianoPose(split='train', train_sec=4)
        
        print("Dataset FPS configuration:")
        
        # Check dataset attributes for FPS
        fps_attrs = ['fps', 'frame_rate', 'frames_per_second', 'temporal_resolution']
        
        for attr in fps_attrs:
            if hasattr(dataset, attr):
                value = getattr(dataset, attr)
                print(f"  Found {attr}: {value}")
        
        # Check a sample to infer FPS
        sample = dataset[0]
        
        if 'position' in sample and 'audio' in sample:
            position_frames = sample['position'].shape[0]
            audio_duration = len(sample['audio']) / 16000  # Assume 16kHz
            
            inferred_fps = position_frames / audio_duration
            print(f"  Inferred FPS from sample: {inferred_fps:.2f}")
            
            # Check if this matches common FPS values
            common_fps = [15, 24, 25, 30, 50, 60]
            closest_fps = min(common_fps, key=lambda x: abs(x - inferred_fps))
            
            if abs(closest_fps - inferred_fps) < 1.0:
                print(f"  Closest standard FPS: {closest_fps}")
                return closest_fps
            else:
                print(f"  Non-standard FPS detected: {inferred_fps:.2f}")
                return inferred_fps
        
        return None
        
    except Exception as e:
        print(f"Could not check dataset FPS configuration: {e}")
        return None
```

### Step 2: Analyze Model Output Frame Count
```python
def analyze_model_output_frame_count():
    """Analyze model output to determine the actual frame rate."""
    
    # Run inference with known audio duration
    test_duration = 4.0  # seconds
    test_audio = np.random.randn(int(test_duration * 16000)).astype(np.float32)
    
    try:
        pose_hat, guide, frame_num = run_model_inference(model, test_audio, device)
        
        print(f"Model output frame analysis:")
        print(f"  Input audio duration: {test_duration:.2f} seconds")
        print(f"  Output frame count: {frame_num}")
        
        # Calculate actual FPS
        actual_fps = frame_num / test_duration
        print(f"  Calculated FPS: {actual_fps:.2f}")
        
        # Check against common frame rates
        common_fps = [15, 24, 25, 30, 50, 60]
        closest_fps = min(common_fps, key=lambda x: abs(x - actual_fps))
        
        print(f"  Closest standard FPS: {closest_fps}")
        
        if abs(closest_fps - actual_fps) < 0.5:
            print(f"  ✓ Model likely uses {closest_fps} FPS")
            return closest_fps
        else:
            print(f"  ⚠ Model uses non-standard FPS: {actual_fps:.2f}")
            return actual_fps
            
    except Exception as e:
        print(f"Could not analyze model output frame count: {e}")
        return None

def test_fps_with_different_durations():
    """Test FPS consistency with different audio durations."""
    
    test_durations = [2.0, 3.0, 4.0]  # Different durations
    calculated_fps = []
    
    print("Testing FPS consistency across different durations:")
    
    for duration in test_durations:
        try:
            test_audio = np.random.randn(int(duration * 16000)).astype(np.float32)
            pose_hat, guide, frame_num = run_model_inference(model, test_audio, device)
            
            fps = frame_num / duration
            calculated_fps.append(fps)
            
            print(f"  Duration {duration:.1f}s: {frame_num} frames → {fps:.2f} FPS")
            
        except Exception as e:
            print(f"  Duration {duration:.1f}s: Failed - {e}")
    
    if calculated_fps:
        mean_fps = np.mean(calculated_fps)
        std_fps = np.std(calculated_fps)
        
        print(f"  Mean FPS: {mean_fps:.2f} ± {std_fps:.2f}")
        
        if std_fps < 0.1:
            print(f"  ✓ Consistent FPS across durations")
            return mean_fps
        else:
            print(f"  ⚠ Inconsistent FPS across durations")
            return None
    
    return None
```

### Step 3: Check Official Rendering Frame Rate
```python
def check_official_rendering_fps():
    """Check the frame rate used in official rendering/visualization."""
    
    try:
        # Check official rendering code
        if os.path.exists('PianoMotion10M/datasets/show.py'):
            with open('PianoMotion10M/datasets/show.py', 'r') as f:
                content = f.read()
                
                print("Official rendering FPS analysis:")
                
                # Look for FPS-related patterns
                fps_patterns = [
                    r'fps\s*=\s*(\d+)',
                    r'frame.*rate\s*=\s*(\d+)',
                    r'(\d+)\s*fps',
                    r'frames.*per.*second\s*=\s*(\d+)'
                ]
                
                for pattern in fps_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  Found FPS in rendering: {matches}")
                
                # Look for video output settings (might indicate FPS)
                if 'cv2.VideoWriter' in content or 'imageio' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'fps' in line.lower() or 'frame' in line.lower():
                            print(f"  Line {i}: {line.strip()}")
        
        # Check if there are example videos with known frame rates
        video_files = []
        for root, dirs, files in os.walk('PianoMotion10M'):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
        
        if video_files:
            print(f"  Found {len(video_files)} video files that could indicate FPS")
            # Could use ffprobe or similar to check video frame rates
        
        return True
        
    except Exception as e:
        print(f"Could not check official rendering FPS: {e}")
        return False

def verify_sequence_length_calculation():
    """Verify that sequence length calculation matches expected values."""
    
    try:
        # Check the model's expected sequence length
        train_sec = 4
        assumed_fps = 30
        expected_seq_length = train_sec * assumed_fps  # 120
        
        print(f"Sequence length verification:")
        print(f"  Training duration: {train_sec} seconds")
        print(f"  Assumed FPS: {assumed_fps}")
        print(f"  Expected sequence length: {expected_seq_length}")
        
        # Check if this matches the model's actual configuration
        if hasattr(model, 'seq_length'):
            actual_seq_length = model.seq_length
            print(f"  Model sequence length: {actual_seq_length}")
            
            if actual_seq_length == expected_seq_length:
                print(f"  ✓ Sequence length matches 30 FPS assumption")
                return True
            else:
                inferred_fps = actual_seq_length / train_sec
                print(f"  ⚠ Sequence length suggests {inferred_fps:.1f} FPS")
                return inferred_fps
        
        return None
        
    except Exception as e:
        print(f"Could not verify sequence length: {e}")
        return None
```

## Recommended Fix

Create a robust FPS detection and configuration system:

```python
def detect_model_fps():
    """Detect the actual FPS used by the model through multiple methods."""
    
    print("=== Model FPS Detection ===")
    
    # Method 1: Check training configuration
    config_fps = verify_model_training_fps()
    
    # Method 2: Check dataset configuration
    dataset_fps = check_dataset_fps_configuration()
    
    # Method 3: Analyze model output
    output_fps = analyze_model_output_frame_count()
    
    # Method 4: Test consistency
    consistent_fps = test_fps_with_different_durations()
    
    # Method 5: Check sequence length
    sequence_fps = verify_sequence_length_calculation()
    
    # Method 6: Check official rendering
    rendering_check = check_official_rendering_fps()
    
    # Synthesize results
    detected_fps = synthesize_fps_detection([
        ('training_config', config_fps),
        ('dataset_config', dataset_fps),
        ('model_output', output_fps),
        ('consistency_test', consistent_fps),
        ('sequence_length', sequence_fps)
    ])
    
    return detected_fps

def synthesize_fps_detection(fps_sources):
    """Synthesize FPS from multiple detection methods."""
    
    print("\nSynthesizing FPS detection results:")
    
    valid_fps = []
    evidence = []
    
    for source_name, fps_value in fps_sources:
        if fps_value and isinstance(fps_value, (int, float)) and fps_value > 0:
            valid_fps.append(fps_value)
            evidence.append(f"{source_name}: {fps_value:.1f}")
            print(f"  {source_name}: {fps_value:.1f} FPS")
        else:
            print(f"  {source_name}: No valid FPS detected")
    
    if not valid_fps:
        print("  ⚠ No valid FPS detected, using default 30 FPS")
        return {
            'fps': 30.0,
            'confidence': 0.0,
            'evidence': ['default fallback'],
            'source': 'fallback'
        }
    
    # Calculate statistics
    mean_fps = np.mean(valid_fps)
    std_fps = np.std(valid_fps) if len(valid_fps) > 1 else 0.0
    
    print(f"\n  Statistics: mean={mean_fps:.2f}, std={std_fps:.2f}")
    
    # Determine confidence based on consistency
    if std_fps < 0.5:
        confidence = 1.0
        final_fps = round(mean_fps)
        source = 'high_confidence_detection'
    elif std_fps < 1.0:
        confidence = 0.7
        final_fps = round(mean_fps)
        source = 'moderate_confidence_detection'
    else:
        confidence = 0.3
        # Use the most common value or default to 30
        final_fps = 30
        source = 'low_confidence_fallback'
    
    result = {
        'fps': float(final_fps),
        'confidence': confidence,
        'evidence': evidence,
        'source': source,
        'raw_values': valid_fps
    }
    
    print(f"\n  Final FPS: {final_fps} (confidence: {confidence:.1f})")
    print(f"  Evidence: {', '.join(evidence)}")
    
    return result

def configure_fps_throughout_codebase(detected_fps_info):
    """Configure FPS consistently throughout the codebase."""
    
    fps = detected_fps_info['fps']
    confidence = detected_fps_info['confidence']
    
    print(f"Configuring FPS throughout codebase: {fps} FPS (confidence: {confidence:.1f})")
    
    # Create a global FPS configuration
    fps_config = {
        'fps': fps,
        'confidence': confidence,
        'samples_per_frame': 16000 / fps,  # For 16kHz audio
        'frames_per_second': fps,
        'detection_info': detected_fps_info
    }
    
    return fps_config

def update_frame_calculations(fps_config):
    """Update all frame-related calculations to use detected FPS."""
    
    fps = fps_config['fps']
    
    # Updated calculation functions
    def calculate_frame_number_updated(audio_length, sample_rate=16000, train_sec=4):
        """Calculate frame number using detected FPS."""
        audio_duration = audio_length / sample_rate
        effective_duration = min(audio_duration, train_sec)
        frame_num = int(effective_duration * fps)
        
        print(f"Frame calculation: {effective_duration:.2f}s × {fps} FPS = {frame_num} frames")
        return frame_num
    
    def generate_timestamps_updated(num_frames):
        """Generate timestamps using detected FPS."""
        return [i / fps for i in range(num_frames)]
    
    def synchronize_audio_with_frames_updated(audio_wave, midi_path, sample_rate=16000):
        """Synchronize audio with frames using detected FPS."""
        audio_duration = len(audio_wave) / sample_rate
        frame_count = int(audio_duration * fps)
        frame_times = np.linspace(0, audio_duration, frame_count)
        
        print(f"Audio synchronization: {audio_duration:.2f}s → {frame_count} frames at {fps} FPS")
        return frame_times, frame_count
    
    return {
        'calculate_frame_number': calculate_frame_number_updated,
        'generate_timestamps': generate_timestamps_updated,
        'synchronize_audio_with_frames': synchronize_audio_with_frames_updated
    }

def validate_fps_configuration(fps_config):
    """Validate the FPS configuration by running test cases."""
    
    fps = fps_config['fps']
    
    print(f"Validating FPS configuration ({fps} FPS):")
    
    # Test 1: Frame count for standard duration
    test_duration = 4.0
    expected_frames = int(test_duration * fps)
    
    try:
        test_audio = np.random.randn(int(test_duration * 16000)).astype(np.float32)
        pose_hat, guide, actual_frames = run_model_inference(model, test_audio, device)
        
        if abs(actual_frames - expected_frames) <= 1:  # Allow 1 frame tolerance
            print(f"  ✓ Frame count validation passed: {actual_frames} ≈ {expected_frames}")
            frame_validation = True
        else:
            print(f"  ⚠ Frame count mismatch: {actual_frames} vs {expected_frames}")
            frame_validation = False
    except Exception as e:
        print(f"  ✗ Frame count validation failed: {e}")
        frame_validation = False
    
    # Test 2: Timing consistency
    test_timestamps = [i / fps for i in range(10)]
    expected_duration = 9 / fps  # 10 frames = 9 intervals
    actual_duration = test_timestamps[-1] - test_timestamps[0]
    
    if abs(actual_duration - expected_duration) < 0.01:
        print(f"  ✓ Timestamp validation passed: {actual_duration:.3f} ≈ {expected_duration:.3f}")
        timing_validation = True
    else:
        print(f"  ⚠ Timestamp validation failed: {actual_duration:.3f} vs {expected_duration:.3f}")
        timing_validation = False
    
    return {
        'frame_count_valid': frame_validation,
        'timing_valid': timing_validation,
        'overall_valid': frame_validation and timing_validation
    }
```

## Integration Instructions

1. **Add FPS detection** at the beginning of the main pipeline:
   ```python
   # In main() of midi_to_frames.py, after model loading:
   print("Detecting model frame rate...")
   fps_info = detect_model_fps()
   fps_config = configure_fps_throughout_codebase(fps_info)
   ```

2. **Replace hardcoded FPS values** with configuration:
   ```python
   # Replace all instances of fps=30 with:
   fps = fps_config['fps']
   
   # Update function calls:
   frame_num = calculate_frame_number(len(audio_wave), fps=fps)
   timestamps = generate_timestamps(frame_num, fps=fps)
   ```

3. **Add FPS metadata** to output:
   ```python
   # In save_output_to_json:
   metadata["fps_configuration"] = fps_config
   metadata["frame_rate"] = fps_config['fps']
   metadata["fps_detection_confidence"] = fps_config['confidence']
   ```

4. **Update visualization functions**:
   ```python
   # Pass FPS to visualization functions for correct animation timing
   def visualize_animation(processed_data, fps=None):
       if fps is None:
           fps = processed_data.get('metadata', {}).get('frame_rate', 30)
       # Use fps for animation timing
   ```

## Testing Strategy

1. **Multi-duration testing** to verify FPS consistency
2. **Cross-validation** with official PianoMotion10M outputs
3. **Timing verification** with known audio-visual pairs
4. **Animation testing** to ensure correct playback speed

## Expected Outcomes

After verification, we expect to:
- Confirm the actual FPS used by PianoMotion10M (likely 30, but could be different)
- Ensure consistent frame rate usage throughout the codebase
- Improve temporal accuracy of all time-based calculations
- Provide confidence metrics for the detected frame rate 