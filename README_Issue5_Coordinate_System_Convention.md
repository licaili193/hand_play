# Issue 5: Coordinate System Axis Convention

## Problem Description

The current implementation assumes a specific coordinate system axis convention without verification:

```python
# Current code in midi_to_frames.py
coord_info = {
    "world_space": {
        "origin": "Piano keyboard center or camera position",
        "x_axis": "Left-right relative to piano keyboard",
        "y_axis": "Up-down (vertical)",
        "z_axis": "Forward-backward (depth from camera/piano)",
        "units": "Scaled units (not meters)",
        "scale_factors": [1.5, 1.5, 25]
    }
}
```

**Issue**: While this coordinate system convention is reasonable, it needs verification against the actual training setup and model expectations. Different coordinate conventions could lead to incorrect spatial interpretations.

## Why This Needs Verification

1. **Spatial Misinterpretation**: Wrong axis convention leads to incorrect understanding of hand movements
2. **Visualization Errors**: 3D visualizations will show movements in wrong directions
3. **Physical Realism**: Hand motions may appear unnatural if axes are interpreted incorrectly
4. **Integration Issues**: Incompatibility with external systems expecting different conventions

## Common Coordinate System Conventions

Different systems use various coordinate conventions:

### 1. **Computer Vision / Camera Conventions**
- **OpenCV**: X-right, Y-down, Z-forward (into scene)
- **Computer Graphics**: X-right, Y-up, Z-out (toward viewer)
- **COLMAP**: X-right, Y-down, Z-forward

### 2. **Robotics Conventions**
- **ROS**: X-forward, Y-left, Z-up
- **Industrial**: X-forward, Y-left, Z-up

### 3. **Piano-Specific Conventions**
- **Keyboard-relative**: X-left/right along keys, Y-up/down above keys, Z-forward/back from keyboard
- **Player-relative**: X-left/right from player view, Y-up/down, Z-forward/back

### 4. **Motion Capture Conventions**
- **Optical**: Various depending on system setup
- **Academic**: Often Y-up, with X and Z varying

## Evidence Analysis

Current assumptions in the code:
- X-axis: "Left-right relative to piano keyboard" 
- Y-axis: "Up-down (vertical)"
- Z-axis: "Forward-backward (depth from camera/piano)"

This suggests a **piano-centric, Y-up convention**, but needs verification.

## Verification Steps

### Step 1: Analyze Training Data Coordinate System
```python
def analyze_training_coordinate_system():
    """Analyze training data to determine coordinate system convention."""
    
    try:
        import sys
        sys.path.append('PianoMotion10M')
        
        # Check dataset implementation
        from datasets.PianoPose import PianoPose
        
        dataset = PianoPose(split='train', train_sec=4)
        sample = dataset[0]
        
        print("Training data coordinate system analysis:")
        
        # Look for coordinate system hints in the dataset
        if 'position' in sample:
            positions = sample['position']
            print(f"  Position data shape: {positions.shape}")
            print(f"  Position range: {np.min(positions, axis=0)} to {np.max(positions, axis=0)}")
            
            # Analyze position distributions
            # Y-axis should show upward bias (hands above piano)
            # X-axis should show spread (left-right movement)
            # Z-axis should show forward bias (hands in front of piano)
            
            pos_stats = {
                'X': {'mean': np.mean(positions[:, 0]), 'std': np.std(positions[:, 0])},
                'Y': {'mean': np.mean(positions[:, 1]), 'std': np.std(positions[:, 1])},
                'Z': {'mean': np.mean(positions[:, 2]), 'std': np.std(positions[:, 2])}
            }
            
            print("  Position statistics:")
            for axis, stats in pos_stats.items():
                print(f"    {axis}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            # Expected patterns for piano playing:
            # - Y should have positive mean (hands above piano surface)
            # - X should have near-zero mean but significant std (lateral movement)
            # - Z should depend on setup (could be positive or negative)
            
            return pos_stats
        
    except Exception as e:
        print(f"Could not analyze training coordinate system: {e}")
        return None

def check_coordinate_system_documentation():
    """Check for coordinate system documentation in the codebase."""
    
    import os
    import re
    
    coordinate_keywords = [
        'coordinate', 'axis', 'x.*axis', 'y.*axis', 'z.*axis',
        'left.*right', 'up.*down', 'forward.*backward',
        'camera.*coordinate', 'world.*coordinate'
    ]
    
    print("Searching for coordinate system documentation...")
    
    for root, dirs, files in os.walk('PianoMotion10M'):
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.rst')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for keyword in coordinate_keywords:
                            matches = re.findall(f'.*{keyword}.*', content, re.IGNORECASE)
                            if matches:
                                print(f"\nFound in {filepath}:")
                                for match in matches[:3]:  # Show first 3 matches
                                    print(f"  {match.strip()}")
                                    
                except:
                    continue
```

### Step 2: Test with Known Movement Patterns
```python
def test_coordinate_convention_with_known_patterns():
    """Test coordinate convention using known piano playing patterns."""
    
    # Run inference with test MIDI
    pose_hat, guide, frame_num = run_model_inference(model, audio_wave, device)
    processed_data = process_model_output_with_proper_coordinates(pose_hat, guide)
    
    # Extract hand positions
    right_pos = processed_data['right_hand_position']
    left_pos = processed_data['left_hand_position']
    
    print("Testing coordinate convention with movement patterns:")
    
    # Test 1: Vertical movement analysis
    right_y_range = np.ptp(right_pos[:, 1])  # Y-axis range
    left_y_range = np.ptp(left_pos[:, 1])
    
    print(f"  Y-axis movement ranges:")
    print(f"    Right hand: {right_y_range:.3f}")
    print(f"    Left hand: {left_y_range:.3f}")
    
    # For piano playing, Y movement should be moderate (hand goes up/down for key strikes)
    reasonable_y_range = (0.05, 0.3)  # 5cm to 30cm vertical movement
    
    if reasonable_y_range[0] <= right_y_range <= reasonable_y_range[1]:
        print(f"    ✓ Right hand Y movement is reasonable for piano playing")
    else:
        print(f"    ⚠ Right hand Y movement seems unusual: {right_y_range:.3f}")
    
    # Test 2: Lateral movement analysis
    combined_x_range = np.ptp(np.concatenate([right_pos[:, 0], left_pos[:, 0]]))
    
    print(f"  X-axis total range: {combined_x_range:.3f}")
    
    # For piano, X should show significant range (hands move across keyboard)
    reasonable_x_range = (0.3, 2.0)  # 30cm to 2m across keyboard
    
    if reasonable_x_range[0] <= combined_x_range <= reasonable_x_range[1]:
        print(f"    ✓ X-axis range is reasonable for keyboard coverage")
    else:
        print(f"    ⚠ X-axis range seems unusual: {combined_x_range:.3f}")
    
    # Test 3: Depth movement analysis
    right_z_range = np.ptp(right_pos[:, 2])
    left_z_range = np.ptp(left_pos[:, 2])
    
    print(f"  Z-axis movement ranges:")
    print(f"    Right hand: {right_z_range:.3f}")
    print(f"    Left hand: {left_z_range:.3f}")
    
    # Z movement should be noticeable but less than X/Y (forward/back motion for dynamics)
    reasonable_z_range = (0.02, 0.5)  # 2cm to 50cm depth movement
    
    # Test 4: Hand separation analysis
    hand_separation = np.mean(np.linalg.norm(right_pos - left_pos, axis=1))
    print(f"  Average hand separation: {hand_separation:.3f}")
    
    # Expected separation for piano playing
    reasonable_separation = (0.1, 1.0)  # 10cm to 1m between hands
    
    return {
        'y_movements': (right_y_range, left_y_range),
        'x_range': combined_x_range,
        'z_movements': (right_z_range, left_z_range),
        'hand_separation': hand_separation
    }

def validate_coordinate_system_with_physics():
    """Validate coordinate system using physical constraints of piano playing."""
    
    # Load a test case and analyze
    pose_hat, guide, frame_num = run_model_inference(model, audio_wave, device)
    processed_data = process_model_output_with_proper_coordinates(pose_hat, guide)
    
    right_pos = processed_data['right_hand_position']
    left_pos = processed_data['left_hand_position']
    
    print("Physical validation of coordinate system:")
    
    # Constraint 1: Hands should generally be above piano (positive Y if Y is up)
    avg_right_y = np.mean(right_pos[:, 1])
    avg_left_y = np.mean(left_pos[:, 1])
    
    print(f"  Average Y positions:")
    print(f"    Right hand: {avg_right_y:.3f}")
    print(f"    Left hand: {avg_left_y:.3f}")
    
    if avg_right_y > 0 and avg_left_y > 0:
        print("    ✓ Hands have positive Y (consistent with Y-up convention)")
        y_up_likely = True
    elif avg_right_y < 0 and avg_left_y < 0:
        print("    ⚠ Hands have negative Y (might indicate Y-down convention)")
        y_up_likely = False
    else:
        print("    ⚠ Mixed Y signs - coordinate system unclear")
        y_up_likely = None
    
    # Constraint 2: Hand positions should be within reasonable piano playing zone
    # Typical piano dimensions: ~1.2m wide, ~0.6m deep, keys at height ~0.7m
    
    all_positions = np.concatenate([right_pos, left_pos], axis=0)
    position_bounds = {
        'X': (np.min(all_positions[:, 0]), np.max(all_positions[:, 0])),
        'Y': (np.min(all_positions[:, 1]), np.max(all_positions[:, 1])),
        'Z': (np.min(all_positions[:, 2]), np.max(all_positions[:, 2]))
    }
    
    print(f"  Position bounds:")
    for axis, (min_val, max_val) in position_bounds.items():
        print(f"    {axis}: [{min_val:.3f}, {max_val:.3f}] (range: {max_val-min_val:.3f})")
    
    # Validate bounds make sense for piano playing
    reasonable_bounds = {
        'X': (-0.8, 0.8),   # ±80cm from center (piano width)
        'Y': (-0.2, 0.4) if not y_up_likely else (0.0, 0.6),  # Height above/below reference
        'Z': (-0.5, 0.5)    # ±50cm depth (forward/back from piano)
    }
    
    bounds_reasonable = True
    for axis, (reasonable_min, reasonable_max) in reasonable_bounds.items():
        actual_min, actual_max = position_bounds[axis]
        if not (reasonable_min <= actual_min and actual_max <= reasonable_max):
            print(f"    ⚠ {axis} bounds outside reasonable range [{reasonable_min}, {reasonable_max}]")
            bounds_reasonable = False
        else:
            print(f"    ✓ {axis} bounds are reasonable")
    
    return {
        'y_up_convention': y_up_likely,
        'bounds_reasonable': bounds_reasonable,
        'position_bounds': position_bounds
    }
```

### Step 3: Cross-Reference with Visualization Code
```python
def analyze_visualization_coordinate_usage():
    """Analyze how visualization code interprets coordinates."""
    
    # Check existing visualization functions
    import inspect
    
    try:
        # Look at 3D visualization functions
        from mano_utils import visualize_hand_3d, set_uniform_axes_3d
        
        # Check axis labeling in visualization
        viz_source = inspect.getsource(visualize_hand_3d)
        
        print("Visualization coordinate usage:")
        
        # Look for axis labels
        if "ax.set_xlabel('X')" in viz_source:
            print("  Found X-axis labeling in visualization")
        if "ax.set_ylabel('Y')" in viz_source:
            print("  Found Y-axis labeling in visualization")
        if "ax.set_zlabel('Z')" in viz_source:
            print("  Found Z-axis labeling in visualization")
        
        # Check for any coordinate transformation or interpretation
        lines = viz_source.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['coord', 'axis', 'transform']):
                print(f"  Line {i}: {line.strip()}")
        
    except Exception as e:
        print(f"Could not analyze visualization coordinate usage: {e}")

def verify_against_official_visualization():
    """Verify coordinate convention against official PianoMotion10M visualization."""
    
    try:
        # Check if official visualization exists
        if os.path.exists('PianoMotion10M/datasets/show.py'):
            with open('PianoMotion10M/datasets/show.py', 'r') as f:
                content = f.read()
                
                print("Official visualization coordinate usage:")
                
                # Look for coordinate system hints
                coord_hints = [
                    'x.*axis', 'y.*axis', 'z.*axis',
                    'coordinate', 'transform', 'position'
                ]
                
                for hint in coord_hints:
                    matches = re.findall(f'.*{hint}.*', content, re.IGNORECASE)
                    if matches:
                        print(f"  Found '{hint}' references:")
                        for match in matches[:2]:
                            print(f"    {match.strip()}")
        
        return True
        
    except Exception as e:
        print(f"Could not verify against official visualization: {e}")
        return False
```

## Recommended Fix

Create a comprehensive coordinate system detection and validation system:

```python
def detect_coordinate_system_convention():
    """Detect the actual coordinate system convention used by the model."""
    
    print("=== Coordinate System Convention Detection ===")
    
    # Step 1: Analyze training data patterns
    training_stats = analyze_training_coordinate_system()
    
    # Step 2: Test with known movement patterns
    movement_analysis = test_coordinate_convention_with_known_patterns()
    
    # Step 3: Validate with physical constraints
    physics_validation = validate_coordinate_system_with_physics()
    
    # Step 4: Cross-reference with official code
    official_confirmed = verify_against_official_visualization()
    
    # Synthesize results
    detected_convention = synthesize_coordinate_convention(
        training_stats, movement_analysis, physics_validation, official_confirmed
    )
    
    return detected_convention

def synthesize_coordinate_convention(training_stats, movement_analysis, physics_validation, official_confirmed):
    """Synthesize coordinate convention from multiple evidence sources."""
    
    print("\nSynthesizing coordinate system convention:")
    
    # Evidence for Y-up vs Y-down
    y_up_evidence = []
    y_down_evidence = []
    
    if physics_validation.get('y_up_convention') is True:
        y_up_evidence.append("Positive average Y positions")
    elif physics_validation.get('y_up_convention') is False:
        y_down_evidence.append("Negative average Y positions")
    
    # Evidence for coordinate interpretation
    interpretation_confidence = 0.0
    interpretation_reasons = []
    
    if movement_analysis:
        if 0.3 <= movement_analysis['x_range'] <= 2.0:
            interpretation_confidence += 0.3
            interpretation_reasons.append("X-range consistent with keyboard width")
        
        y_ranges = movement_analysis['y_movements']
        if all(0.05 <= y_range <= 0.3 for y_range in y_ranges):
            interpretation_confidence += 0.2
            interpretation_reasons.append("Y-movements consistent with key strikes")
        
        if 0.1 <= movement_analysis['hand_separation'] <= 1.0:
            interpretation_confidence += 0.2
            interpretation_reasons.append("Hand separation reasonable")
    
    if physics_validation.get('bounds_reasonable'):
        interpretation_confidence += 0.3
        interpretation_reasons.append("Position bounds physically reasonable")
    
    # Determine final convention
    if len(y_up_evidence) > len(y_down_evidence):
        y_convention = "Y-up"
    elif len(y_down_evidence) > len(y_up_evidence):
        y_convention = "Y-down"
    else:
        y_convention = "Y-up (default)"  # Conservative assumption
    
    detected_convention = {
        'x_axis': 'Left-right relative to piano keyboard',
        'y_axis': f'Vertical ({y_convention})',
        'z_axis': 'Forward-backward (depth from piano)',
        'confidence': interpretation_confidence,
        'evidence': interpretation_reasons,
        'y_convention': y_convention
    }
    
    print(f"  Detected convention:")
    print(f"    X-axis: {detected_convention['x_axis']}")
    print(f"    Y-axis: {detected_convention['y_axis']}")
    print(f"    Z-axis: {detected_convention['z_axis']}")
    print(f"    Confidence: {detected_convention['confidence']:.2f}")
    print(f"    Evidence: {', '.join(detected_convention['evidence'])}")
    
    return detected_convention

def apply_coordinate_system_corrections(processed_data, detected_convention):
    """Apply any necessary coordinate system corrections."""
    
    print(f"Applying coordinate system corrections...")
    
    corrections_applied = []
    
    # Check if Y-axis needs flipping
    if detected_convention['y_convention'] == 'Y-down':
        print("  Flipping Y-axis to match Y-up convention")
        
        # Flip Y coordinates
        processed_data['right_hand_position'][:, 1] *= -1
        processed_data['left_hand_position'][:, 1] *= -1
        
        corrections_applied.append("Y-axis flipped")
    
    # Add coordinate system metadata
    processed_data['coordinate_system'] = {
        'convention': detected_convention,
        'corrections_applied': corrections_applied,
        'verification_confidence': detected_convention['confidence']
    }
    
    print(f"  Corrections applied: {corrections_applied}")
    
    return processed_data

def validate_coordinate_system_with_visualization(processed_data):
    """Validate coordinate system by creating test visualizations."""
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print("Creating coordinate system validation visualization...")
    
    # Create a simple 3D plot to verify axis orientations
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hand positions
    right_pos = processed_data['right_hand_position']
    left_pos = processed_data['left_hand_position']
    
    ax.scatter(right_pos[:, 0], right_pos[:, 1], right_pos[:, 2], 
              c='red', label='Right Hand', alpha=0.6)
    ax.scatter(left_pos[:, 0], left_pos[:, 1], left_pos[:, 2], 
              c='blue', label='Left Hand', alpha=0.6)
    
    # Add coordinate system reference
    origin = np.mean(np.concatenate([right_pos, left_pos]), axis=0)
    
    # Draw coordinate axes
    axis_length = 0.1
    ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, 
             color='red', arrow_length_ratio=0.1, label='X (Left-Right)')
    ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, 
             color='green', arrow_length_ratio=0.1, label='Y (Up-Down)')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, 
             color='blue', arrow_length_ratio=0.1, label='Z (Forward-Back)')
    
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Up-Down)')
    ax.set_zlabel('Z (Forward-Back)')
    ax.set_title('Coordinate System Validation')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('coordinate_system_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Coordinate system validation plot saved as 'coordinate_system_validation.png'")
```

## Integration Instructions

1. **Add coordinate system detection** to the main processing pipeline:
   ```python
   # In process_model_output_with_proper_coordinates:
   detected_convention = detect_coordinate_system_convention()
   processed_data = apply_coordinate_system_corrections(processed_data, detected_convention)
   ```

2. **Update visualization functions** to use detected convention:
   ```python
   # In visualization functions, use the detected axis labels
   convention = processed_data.get('coordinate_system', {}).get('convention', {})
   x_label = convention.get('x_axis', 'X')
   y_label = convention.get('y_axis', 'Y') 
   z_label = convention.get('z_axis', 'Z')
   ```

3. **Add metadata tracking**:
   ```python
   # In save_output_to_json:
   metadata["coordinate_system"] = processed_data.get('coordinate_system', {})
   ```

## Testing Strategy

1. **Multi-MIDI testing** with different musical patterns
2. **Visual verification** of coordinate system orientation
3. **Cross-reference** with official PianoMotion10M examples
4. **Physical validation** against known piano playing constraints

## Expected Outcomes

After verification, we expect to:
- Confirm the correct coordinate system convention
- Detect any necessary axis corrections
- Ensure spatial interpretations are physically realistic
- Provide confidence metrics for the detected convention 