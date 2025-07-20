# Issue 3: Incorrect Model Sample Method Usage

## Problem Description

The current implementation in `midi_to_frames.py` uses an incorrect method signature when calling the diffusion model's `sample` method. This leads to incorrect inference behavior and potentially wrong outputs.

### Specific Issues:

1. **Wrong Method Signature**:
   - Current code: `pose_hat, guide = model.sample(audio_tensor, frame_num, 1)`
   - Official code: `pose_hat, guide = model.sample(audio, frame_num, right_pose.shape[0])`
   - The third parameter should be batch size from actual data, not hardcoded `1`

2. **Missing Proper Audio Format**:
   - Current code passes audio directly without proper preprocessing
   - Official code uses preprocessed audio from the dataset
   - Missing proper audio format validation

3. **Incorrect Frame Number Calculation**:
   - Current code forces fixed frame count: `seq_length = diffusion_args.train_sec * 30`
   - Official code uses dynamic frame count based on actual audio/data length
   - This can cause mismatches between audio and expected output length

4. **Missing Dummy Pose Handling**:
   - Current code creates dummy guide tensors but doesn't use them properly
   - Official inference relies on the model's internal guide generation
   - Missing proper conditioning setup

### Code Location:
- File: `midi_to_frames.py`
- Lines: 135-145 (model inference section)

## Recommended Fixes

### 1. Understand the Correct Sample Method

Based on the official `infer.py`, the correct usage pattern is:

```python
# From official PianoMotion10M/infer.py:
batch, para = valid_dataset.__getitem__(test_id, True)
for key in batch.keys():
    batch[key] = torch.tensor(batch[key]).cuda().unsqueeze(0)

audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']
frame_num = left_pose.shape[1]  # Dynamic frame count from data

# Correct sample call:
pose_hat, guide = model.sample(audio, frame_num, right_pose.shape[0])
```

### 2. Fix Audio Preprocessing

Add proper audio preprocessing to match the expected format:

```python
def preprocess_audio_for_inference(audio_wave, target_duration=4.0, sample_rate=16000):
    """Preprocess audio to match model expectations."""
    
    target_samples = int(target_duration * sample_rate)
    
    # Trim or pad audio to target duration
    if len(audio_wave) > target_samples:
        # Trim to target duration
        audio_wave = audio_wave[:target_samples]
        print(f"✓ Audio trimmed to {target_duration} seconds")
    elif len(audio_wave) < target_samples:
        # Pad with zeros
        padding = target_samples - len(audio_wave)
        audio_wave = np.pad(audio_wave, (0, padding), mode='constant')
        print(f"✓ Audio padded to {target_duration} seconds")
    
    # Ensure proper normalization
    max_val = np.max(np.abs(audio_wave))
    if max_val > 0:
        audio_wave = audio_wave / max_val
    
    # Convert to tensor with proper dimensions
    audio_tensor = torch.tensor(audio_wave, dtype=torch.float32).unsqueeze(0)  # (1, samples)
    
    return audio_tensor, len(audio_wave)
```

### 3. Fix Frame Number Calculation

Implement proper frame number calculation based on audio duration:

```python
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
```

### 4. Implement Correct Model Sample Call

Replace the current inference code with:

```python
def run_model_inference(model, audio_wave, device, train_sec=4):
    """Run model inference with correct method signature."""
    
    try:
        # Preprocess audio properly
        audio_tensor, audio_length = preprocess_audio_for_inference(
            audio_wave, target_duration=train_sec
        )
        audio_tensor = audio_tensor.to(device)
        
        # Calculate proper frame number
        frame_num = calculate_frame_number(audio_length, train_sec=train_sec)
        
        # Validate inputs
        print(f"Input validation:")
        print(f"  - Audio tensor shape: {audio_tensor.shape}")
        print(f"  - Frame number: {frame_num}")
        print(f"  - Device: {device}")
        
        with torch.no_grad():
            # Use correct sample method signature
            # Note: batch_size = 1 for single inference
            batch_size = 1
            pose_hat, guide = model.sample(audio_tensor, frame_num, batch_size)
            
            # Ensure correct output dimensions
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
```

### 5. Add Model Sample Method Validation

Create validation to ensure the sample method works correctly:

```python
def validate_sample_method(model, audio_tensor, frame_num, batch_size):
    """Validate that the model sample method works with given parameters."""
    
    try:
        # Test the sample method signature
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
```

### 6. Complete Fixed Inference Section

Replace the current inference section with:

```python
# 3. Run inference
print("\nStep 3: Running inference on audio...")
try:
    # Validate sample method before running inference
    audio_test = torch.zeros(1, 1000).to(device)  # Small test tensor
    if not validate_sample_method(model, audio_test, 10, 1):
        print("✗ Model sample method validation failed")
        return
    
    # Run actual inference
    pose_hat, guide, actual_frame_num = run_model_inference(
        model, audio_wave, device, train_sec=diffusion_args.train_sec
    )
    
    # Validate output dimensions
    validate_model_output(pose_hat, guide)
    
    print(f"✓ Model inference completed:")
    print(f"  - Processed {actual_frame_num} frames")
    print(f"  - Duration: {actual_frame_num / 30:.2f} seconds")
    
except Exception as e:
    print(f"✗ Error during inference: {e}")
    return
```

### 7. Handle Audio-Frame Synchronization

Add proper synchronization between audio length and frame count:

```python
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
```

## Testing the Fix

1. **Method signature validation**: Confirm the sample method accepts the correct parameters
2. **Audio format testing**: Verify audio preprocessing produces expected tensor format
3. **Frame synchronization**: Check that audio length matches frame count expectations
4. **Output dimension validation**: Ensure outputs have expected shapes
5. **Comparison with official**: Test against official inference patterns

## Impact of Fix

- **Correct model usage**: Sample method is called with proper parameters
- **Proper audio handling**: Audio is preprocessed to match model expectations
- **Accurate frame counting**: Frame numbers are calculated based on actual audio duration
- **Better error handling**: Validation catches issues before inference fails
- **Consistency with official code**: Matches the official inference patterns and behavior 