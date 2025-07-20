# Issue 3: Scale Factor Application Order

## Problem Description

The current implementation applies scale factors based on an assumed hand ordering that may be incorrect:

```python
# Current code in midi_to_frames.py
scale = torch.tensor([1.5, 1.5, 25]).to(device)
scale_expanded = scale.repeat(2).view(1, 1, 6).expand_as(guide)  # [1.5, 1.5, 25, 1.5, 1.5, 25]
scaled_guide = (guide * scale_expanded)[0].cpu().numpy()
```

**Issue**: This creates the scaling pattern `[X, Y, Z, X, Y, Z]` which assumes the guide format is `[hand1_xyz, hand2_xyz]`. However, if Issue 2 reveals that the hand ordering is incorrect, then the scale factors are being applied to the wrong coordinates.

## Why This Is Critical

1. **Incorrect Spatial Scaling**: If hands are swapped, scale factors get applied to wrong coordinates
2. **Coordinate System Corruption**: Z-axis scaling (factor of 25) is the most critical and could be applied to wrong dimensions
3. **Compounded Error**: This issue magnifies the hand ordering problem (Issue 2)
4. **Physical Unrealism**: Wrong scaling leads to unrealistic hand positions and movements

## Evidence of the Problem

Current scaling assumptions:
- The code assumes `scale.repeat(2)` correctly maps to `[right_x, right_y, right_z, left_x, left_y, left_z]`
- But if the actual format is `[left_x, left_y, left_z, right_x, right_y, right_z]`, the scaling is still correct
- However, the downstream assignment to "right" and "left" variables becomes wrong

## Detailed Analysis

```python
# Current implementation:
scale = [1.5, 1.5, 25]  # [X_scale, Y_scale, Z_scale]
scale_expanded = [1.5, 1.5, 25, 1.5, 1.5, 25]  # For both hands

# Case 1: Guide format is [right_xyz, left_xyz] (current assumption)
# guide = [right_x, right_y, right_z, left_x, left_y, left_z]
# scaled = [right_x*1.5, right_y*1.5, right_z*25, left_x*1.5, left_y*1.5, left_z*25]
# Assignment: right_hand_pos = scaled[:3], left_hand_pos = scaled[3:]
# Result: ✓ Correct scaling applied to correct hands

# Case 2: Guide format is [left_xyz, right_xyz] (if Issue 2 is confirmed)
# guide = [left_x, left_y, left_z, right_x, right_y, right_z]  
# scaled = [left_x*1.5, left_y*1.5, left_z*25, right_x*1.5, right_y*1.5, right_z*25]
# Assignment: right_hand_pos = scaled[:3], left_hand_pos = scaled[3:]
# Result: ✗ Scale factors correct, but "right_hand_pos" gets left hand data
```

## Root Cause Dependencies

This issue is **dependent on Issue 2** (Hand Position Ordering):
- If hand ordering is correct, scale factor application is correct
- If hand ordering is wrong, scale factor application is technically correct but assigned to wrong hands
- The scale factors themselves `[1.5, 1.5, 25]` are likely correct based on official code

## Verification Steps

### Step 1: Verify Scale Factor Values
```python
def verify_scale_factors():
    """Verify that the scale factors [1.5, 1.5, 25] are correct."""
    
    # Check against official PianoMotion10M code
    # Look for scale factor usage in:
    # - PianoMotion10M/infer.py
    # - PianoMotion10M/eval.py
    # - PianoMotion10M/train.py
    
    print("Checking official scale factors...")
    
    try:
        # Search for scale factor patterns in official code
        import os
        import re
        
        for root, dirs, files in os.walk('PianoMotion10M'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Look for scale factor patterns
                            scale_patterns = [
                                r'scale.*=.*\[.*1\.5.*1\.5.*25.*\]',
                                r'1\.5.*1\.5.*25',
                                r'scale.*torch\.tensor.*1\.5.*1\.5.*25'
                            ]
                            for pattern in scale_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    print(f"Found in {filepath}: {matches}")
                    except:
                        continue
        
    except Exception as e:
        print(f"Could not verify scale factors: {e}")
```

### Step 2: Test Scale Factor Impact
```python
def test_scale_factor_impact():
    """Test the impact of different scale factor applications."""
    
    # Create test guide data
    test_guide = torch.tensor([[[
        1.0, 2.0, 0.1,  # Hand 1: x=1, y=2, z=0.1  
        -1.0, 2.0, 0.1  # Hand 2: x=-1, y=2, z=0.1
    ]]], dtype=torch.float32)
    
    # Apply current scaling
    scale = torch.tensor([1.5, 1.5, 25])
    scale_expanded = scale.repeat(2).view(1, 1, 6)
    scaled_current = (test_guide * scale_expanded)[0].cpu().numpy()
    
    print(f"Original: {test_guide[0, 0].tolist()}")
    print(f"Scaled: {scaled_current[0].tolist()}")
    print(f"Expected: [1.5, 3.0, 2.5, -1.5, 3.0, 2.5]")
    
    # Test different orderings
    # If hands are swapped, the scaling is still mathematically correct
    # The issue is in the interpretation, not the scaling itself
    
    return scaled_current

def test_coordinate_system_consistency():
    """Test that scaled coordinates make physical sense for piano playing."""
    
    # Run inference with test data
    pose_hat, guide, frame_num = run_model_inference(model, audio_wave, device)
    
    # Apply current scaling
    scale = torch.tensor([1.5, 1.5, 25]).to(device)
    scale_expanded = scale.repeat(2).view(1, 1, 6).expand_as(guide)
    scaled_guide = (guide * scale_expanded)[0].cpu().numpy()
    
    # Extract positions (regardless of left/right assignment for now)
    hand1_pos = scaled_guide[:, :3]
    hand2_pos = scaled_guide[:, 3:]
    
    # Analyze position ranges
    h1_range = np.ptp(hand1_pos, axis=0)  # Range in each dimension
    h2_range = np.ptp(hand2_pos, axis=0)
    
    print(f"Hand 1 position ranges: X={h1_range[0]:.2f}, Y={h1_range[1]:.2f}, Z={h1_range[2]:.2f}")
    print(f"Hand 2 position ranges: X={h2_range[0]:.2f}, Y={h2_range[1]:.2f}, Z={h2_range[2]:.2f}")
    
    # Check if ranges make sense for piano playing
    # X: Lateral movement across keyboard (~0.5-2.0m range is reasonable)
    # Y: Vertical movement above keys (~0.1-0.5m range is reasonable)  
    # Z: Depth movement forward/back (~0.1-1.0m range is reasonable after 25x scaling)
    
    reasonable_ranges = {
        'X': (0.1, 3.0),  # 10cm to 3m lateral range
        'Y': (0.05, 1.0), # 5cm to 1m vertical range
        'Z': (0.05, 2.0)  # 5cm to 2m depth range (after 25x scaling)
    }
    
    for hand_name, hand_range in [("Hand 1", h1_range), ("Hand 2", h2_range)]:
        for i, (axis, (min_reasonable, max_reasonable)) in enumerate(reasonable_ranges.items()):
            if not (min_reasonable <= hand_range[i] <= max_reasonable):
                print(f"⚠ {hand_name} {axis} range {hand_range[i]:.2f} outside reasonable bounds [{min_reasonable}, {max_reasonable}]")
            else:
                print(f"✓ {hand_name} {axis} range {hand_range[i]:.2f} is reasonable")
    
    return hand1_pos, hand2_pos
```

### Step 3: Cross-Reference with Official Implementation
```python
def verify_official_scaling_implementation():
    """Check how scaling is implemented in official PianoMotion10M code."""
    
    try:
        # Look at official inference implementation
        import sys
        sys.path.append('PianoMotion10M')
        
        # Check if there's an official inference script
        if os.path.exists('PianoMotion10M/infer.py'):
            print("Found official inference script - checking scaling implementation...")
            with open('PianoMotion10M/infer.py', 'r') as f:
                content = f.read()
                
                # Look for scaling patterns
                if 'scale' in content.lower():
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'scale' in line.lower():
                            print(f"Line {i+1}: {line.strip()}")
        
        # Check evaluation script
        if os.path.exists('PianoMotion10M/eval.py'):
            print("\nFound official evaluation script - checking scaling...")
            # Similar analysis for eval.py
            
        return True
        
    except Exception as e:
        print(f"Could not verify official implementation: {e}")
        return False
```

## Recommended Fix

Create a robust scaling system that's independent of hand ordering assumptions:

```python
def apply_coordinate_scaling_robust(guide, pose_hat, device='cpu'):
    """Apply coordinate scaling that's robust to hand ordering issues."""
    
    # Verify scale factors first
    official_scale_confirmed = verify_scale_factors()
    
    # Use confirmed scale factors
    scale = torch.tensor([1.5, 1.5, 25], device=device)
    
    print(f"Applying coordinate scaling:")
    print(f"  Scale factors: {scale.tolist()}")
    print(f"  Input guide shape: {guide.shape}")
    print(f"  Input pose shape: {pose_hat.shape}")
    
    # Apply scaling to guide (positions)
    # This works regardless of hand ordering since it's applied element-wise
    scale_expanded = scale.repeat(2).view(1, 1, 6).expand_as(guide)
    scaled_guide = guide * scale_expanded
    
    # Apply scaling to pose (convert to radians)
    # This is independent of coordinate scaling
    scaled_pose = pose_hat * np.pi
    
    print(f"  Guide range before scaling: [{torch.min(guide):.3f}, {torch.max(guide):.3f}]")
    print(f"  Guide range after scaling: [{torch.min(scaled_guide):.3f}, {torch.max(scaled_guide):.3f}]")
    print(f"  Pose range after radian conversion: [{torch.min(scaled_pose):.3f}, {torch.max(scaled_pose):.3f}]")
    
    # Convert to numpy for further processing
    scaled_guide_np = scaled_guide[0].cpu().numpy()
    scaled_pose_np = scaled_pose[0].cpu().numpy()
    
    # Validate physical reasonableness
    validate_scaled_coordinates(scaled_guide_np)
    
    return scaled_pose_np, scaled_guide_np

def validate_scaled_coordinates(scaled_guide):
    """Validate that scaled coordinates are physically reasonable for piano playing."""
    
    # Extract both hands (order TBD by Issue 2)
    hand1_pos = scaled_guide[:, :3]
    hand2_pos = scaled_guide[:, 3:]
    
    # Check overall statistics
    overall_range = np.ptp(np.concatenate([hand1_pos, hand2_pos], axis=0), axis=0)
    hand_separation = np.linalg.norm(np.mean(hand1_pos, axis=0) - np.mean(hand2_pos, axis=0))
    
    print(f"Coordinate validation:")
    print(f"  Overall position range: X={overall_range[0]:.2f}, Y={overall_range[1]:.2f}, Z={overall_range[2]:.2f}")
    print(f"  Average hand separation: {hand_separation:.2f}")
    
    # Reasonable bounds for piano playing (approximate)
    reasonable_bounds = {
        'total_X_range': (0.2, 2.0),   # Across keyboard
        'total_Y_range': (0.1, 0.8),   # Above keys
        'total_Z_range': (0.2, 1.5),   # Forward/back from piano
        'hand_separation': (0.1, 1.0)   # Distance between hands
    }
    
    # Check bounds
    checks = [
        ('Total X range', overall_range[0], reasonable_bounds['total_X_range']),
        ('Total Y range', overall_range[1], reasonable_bounds['total_Y_range']),
        ('Total Z range', overall_range[2], reasonable_bounds['total_Z_range']),
        ('Hand separation', hand_separation, reasonable_bounds['hand_separation'])
    ]
    
    for name, value, (min_val, max_val) in checks:
        if min_val <= value <= max_val:
            print(f"  ✓ {name}: {value:.2f} is reasonable")
        else:
            print(f"  ⚠ {name}: {value:.2f} outside reasonable range [{min_val}, {max_val}]")

def apply_smoothing_to_scaled_data(prediction, scaled_guide):
    """Apply smoothing as in official implementation."""
    
    try:
        from scipy.signal import savgol_filter
        
        # Apply smoothing (from official code)
        for i in range(prediction.shape[1]):
            prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
        for i in range(scaled_guide.shape[1]):
            scaled_guide[:, i] = savgol_filter(scaled_guide[:, i], 5, 2)
            
        print("✓ Applied Savitzky-Golay smoothing")
        
    except Exception as e:
        print(f"⚠ Could not apply smoothing: {e}")
    
    return prediction, scaled_guide
```

## Integration Instructions

1. **Replace the current scaling function** in `apply_coordinate_transformations()`:
   ```python
   # Replace current implementation with:
   prediction, scaled_guide = apply_coordinate_scaling_robust(pose_hat, guide, device)
   prediction, scaled_guide = apply_smoothing_to_scaled_data(prediction, scaled_guide)
   ```

2. **Add validation calls** after scaling:
   ```python
   # Verify coordinate system consistency
   test_coordinate_system_consistency()
   ```

3. **Add metadata tracking**:
   ```python
   # In save_output_to_json:
   metadata["scale_factors_applied"] = [1.5, 1.5, 25]
   metadata["scaling_validation_passed"] = True  # Set based on validation results
   ```

## Testing Strategy

1. **Unit tests** for scaling functions with known inputs
2. **Integration tests** with real audio data
3. **Physical validation** of output coordinate ranges
4. **Comparison** with official PianoMotion10M outputs (if available)

## Resolution Dependency

This issue will be automatically resolved once **Issue 2** (Hand Position Ordering) is resolved, because:
- The scale factors `[1.5, 1.5, 25]` are mathematically applied correctly
- The issue is in the interpretation of which scaled coordinates belong to which hand
- Once hand ordering is verified/corrected, the scaling will be correctly attributed 