# Issue 1: Model Output Dimension Uncertainty

## Problem Description

The current implementation makes assumptions about the model output tensor dimensions without verification:

```python
# Current code in midi_to_frames.py
if pose_hat.dim() == 3:
    pose_hat = pose_hat.permute(0, 2, 1)  # (batch, frames, 96)
if guide.dim() == 3:
    guide = guide.permute(0, 2, 1)        # (batch, frames, 6)
```

**Issue**: The code assumes the model outputs tensors in format `(batch, channels, frames)` and automatically permutes them to `(batch, frames, channels)`. However, different PyTorch model implementations can output different dimension orders, and this assumption may be incorrect.

## Why This Is Critical

1. **Data Corruption**: If the model actually outputs `(batch, frames, channels)`, the permutation will incorrectly swap time and feature dimensions
2. **Silent Failure**: The code may run without errors but produce meaningless results
3. **Downstream Effects**: All subsequent processing (scaling, MANO conversion, visualization) will be wrong

## Evidence of the Problem

From the codebase:
- The model uses transformer architecture which commonly outputs `(batch, seq_len, features)`
- The validation step doesn't verify the actual output format
- The permutation is applied conditionally based on number of dimensions, not content verification

## Verification Steps

### Step 1: Inspect Model Architecture
```python
# Add this after model loading in midi_to_frames.py
print("=== Model Architecture Inspection ===")
print(f"Piano2Posi output dim: {model.piano2posi.bs_dim}")  # Should be 6
print(f"Diffusion output dim: {model.unet.channels}")       # Should be 96

# Test with dummy input to see raw output format
with torch.no_grad():
    dummy_audio = torch.randn(1, 64000).to(device)  # 4 seconds
    raw_pose, raw_guide = model.sample(dummy_audio, 120, 1)
    print(f"Raw pose output shape: {raw_pose.shape}")
    print(f"Raw guide output shape: {raw_guide.shape}")
```

### Step 2: Content-Based Validation
```python
# Add dimension validation before permutation
def validate_model_output_format(pose_hat, guide):
    """Validate model output format by checking content patterns."""
    
    print(f"Original shapes - pose: {pose_hat.shape}, guide: {guide.shape}")
    
    # Test both possible interpretations
    if pose_hat.dim() == 3:
        # Option A: (batch, channels, frames) - needs permutation
        option_a_pose = pose_hat.permute(0, 2, 1)
        # Option B: (batch, frames, channels) - no permutation needed
        option_b_pose = pose_hat
        
        print(f"Option A (after permute): {option_a_pose.shape}")
        print(f"Option B (no permute): {option_b_pose.shape}")
        
        # Check which makes more sense based on expected dimensions
        # Guide should be (batch, frames, 6)
        # Pose should be (batch, frames, 96)
        
        if option_a_pose.shape[-1] == 96 and guide.shape[-1] == 6:
            print("✓ Option A appears correct: (batch, channels, frames) → (batch, frames, channels)")
            return option_a_pose, guide
        elif option_b_pose.shape[-1] == 96 and guide.shape[-1] == 6:
            print("✓ Option B appears correct: already (batch, frames, channels)")
            return option_b_pose, guide
        else:
            print("✗ Neither option produces expected dimensions!")
            print(f"Expected: pose[..., 96], guide[..., 6]")
            print(f"Got A: pose[..., {option_a_pose.shape[-1]}], guide[..., {guide.shape[-1]}]")
            print(f"Got B: pose[..., {option_b_pose.shape[-1]}], guide[..., {guide.shape[-1]}]")
            raise ValueError("Model output dimensions don't match expected format")
    
    return pose_hat, guide
```

### Step 3: Check Against Official Implementation
```python
# Look for official inference code in PianoMotion10M repository
# File: PianoMotion10M/infer.py or similar
# Compare dimension handling with official implementation
```

## Recommended Fix

Replace the current permutation logic with validated dimension handling:

```python
def handle_model_output_dimensions(pose_hat, guide):
    """Handle model output dimensions with proper validation."""
    
    # Validate basic requirements
    assert pose_hat.shape[0] == guide.shape[0], "Batch size mismatch"
    assert pose_hat.shape[1] == guide.shape[1], "Sequence length mismatch" 
    
    # Expected final shapes: (batch, frames, 96) and (batch, frames, 6)
    target_pose_features = 96
    target_guide_features = 6
    
    # Check current format and adjust if needed
    if pose_hat.shape[-1] == target_pose_features and guide.shape[-1] == target_guide_features:
        # Already in correct format: (batch, frames, channels)
        print("✓ Model outputs already in (batch, frames, channels) format")
        return pose_hat, guide
        
    elif pose_hat.shape[1] == target_pose_features and guide.shape[1] == target_guide_features:
        # Need permutation: (batch, channels, frames) → (batch, frames, channels)
        print("✓ Permuting from (batch, channels, frames) to (batch, frames, channels)")
        return pose_hat.permute(0, 2, 1), guide.permute(0, 2, 1)
        
    else:
        # Unexpected format
        raise ValueError(
            f"Unexpected model output dimensions:\n"
            f"Pose: {pose_hat.shape} (expected: (batch, frames, 96) or (batch, 96, frames))\n"
            f"Guide: {guide.shape} (expected: (batch, frames, 6) or (batch, 6, frames))"
        )

# Replace the current permutation code with:
pose_hat, guide = handle_model_output_dimensions(pose_hat, guide)
```

## Testing the Fix

```python
# Test with known input to verify correct behavior
def test_dimension_handling():
    # Create test tensors in both possible formats
    batch_size, frames, pose_dim, guide_dim = 1, 120, 96, 6
    
    # Format A: (batch, channels, frames)
    test_pose_a = torch.randn(batch_size, pose_dim, frames)
    test_guide_a = torch.randn(batch_size, guide_dim, frames)
    
    # Format B: (batch, frames, channels) 
    test_pose_b = torch.randn(batch_size, frames, pose_dim)
    test_guide_b = torch.randn(batch_size, frames, guide_dim)
    
    # Test both formats
    result_a = handle_model_output_dimensions(test_pose_a, test_guide_a)
    result_b = handle_model_output_dimensions(test_pose_b, test_guide_b)
    
    # Both should result in (batch, frames, channels)
    assert result_a[0].shape == (batch_size, frames, pose_dim)
    assert result_a[1].shape == (batch_size, frames, guide_dim)
    assert result_b[0].shape == (batch_size, frames, pose_dim)
    assert result_b[1].shape == (batch_size, frames, guide_dim)
    
    print("✓ Dimension handling test passed")

test_dimension_handling()
```

## Integration Instructions

1. Add the validation function to `midi_to_frames.py`
2. Replace the current permutation logic in `run_model_inference()`
3. Add the test function and run it during development
4. Monitor output shapes in all downstream functions to ensure consistency 