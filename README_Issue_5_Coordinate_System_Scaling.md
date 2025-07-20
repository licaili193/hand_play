# Issue 5: Coordinate System and Scaling Issues

## Problem Description

The current implementation in `midi_to_frames.py` and `mano_utils.py` has significant issues with coordinate system handling and scaling factors that cause incorrect spatial representation of hand motions.

### Specific Issues:

1. **Missing Critical Scaling Factors**:
   - Official code applies scaling: `scale = torch.tensor([1.5, 1.5, 25])`
   - Guide positions need scaling: `guide * scale.repeat(2)` for both hands
   - Current code doesn't apply any scaling, leading to incorrect spatial coordinates

2. **Incorrect Coordinate System Assumptions**:
   - Current code assumes direct 3D joint positions
   - Model outputs are in a specific coordinate frame that needs transformation
   - Missing understanding of the world coordinate system used in training

3. **Wrong Visualization Coordinate Frame**:
   - MANO utilities assume standard MANO coordinate system
   - Model outputs may be in a different coordinate frame (camera/world space)
   - Need proper coordinate transformations for visualization

4. **Missing Spatial Context**:
   - No understanding of the physical scale of hand motions
   - Piano keyboard spatial relationship not considered
   - Hand positions relative to piano keys not properly handled

### Code Location:
- File: `midi_to_frames.py`
- Lines: 145-165 (output processing without scaling)
- File: `mano_utils.py`
- Lines: 350-400 (visualization functions)

## Recommended Fixes

### 1. Implement Proper Scaling and Coordinate Transformation

Add comprehensive coordinate system handling:

```python
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
    prediction = pose_hat[0].detach().cpu().numpy() * np.pi
    print(f"  - Converted pose to radians (range: [{np.min(prediction):.3f}, {np.max(prediction):.3f}])")
    
    # Apply scaling to guide positions (hand positions in world space)
    # scale.repeat(2) creates [1.5, 1.5, 25, 1.5, 1.5, 25] for both hands
    scaled_guide = (guide * scale.repeat(2))[0].cpu().numpy()
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
```

### 2. Fix Spatial Scale Understanding

Implement proper spatial scale interpretation:

```python
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
    right_physical = right_range * np.array([estimated_scale['x'], estimated_scale['y'], estimated_scale['z']])
    left_physical = left_range * np.array([estimated_scale['x'], estimated_scale['y'], estimated_scale['z']])
    
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
```

### 3. Add Piano Keyboard Spatial Context

Implement piano keyboard spatial relationship:

```python
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
    estimated_octaves = total_hand_span / (keyboard_info['approximate_octave_width'] / estimated_scale['x'])
    print(f"  Estimated keyboard coverage: ~{estimated_octaves:.1f} octaves")
    
    return keyboard_info
```

### 4. Fix Visualization Coordinate System

Update visualization to use correct coordinate transformations:

```python
def visualize_with_correct_coordinates(processed_data, frame_idx=0):
    """
    Visualize hand motion with correct coordinate system handling.
    """
    
    # Get coordinate system information
    coord_info = understand_coordinate_system()
    
    # Extract data for visualization
    right_pos = processed_data['right_hand_position'][frame_idx]
    left_pos = processed_data['left_hand_position'][frame_idx]
    right_angles = processed_data['right_hand_angles'][frame_idx]
    left_angles = processed_data['left_hand_angles'][frame_idx]
    
    print(f"Visualization Frame {frame_idx}:")
    print(f"  Right hand position: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
    print(f"  Left hand position: [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
    print(f"  Right hand angles range: [{np.min(right_angles):.3f}, {np.max(right_angles):.3f}] rad")
    print(f"  Left hand angles range: [{np.min(left_angles):.3f}, {np.max(left_angles):.3f}] rad")
    
    # Create 3D visualization with proper coordinate system
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Hand positions in world space
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(right_pos[0], right_pos[1], right_pos[2], 
               c='red', s=100, label='Right Hand', alpha=0.8)
    ax1.scatter(left_pos[0], left_pos[1], left_pos[2], 
               c='blue', s=100, label='Left Hand', alpha=0.8)
    
    ax1.set_xlabel('X (Lateral)')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('Hand Positions\n(Scaled World Coordinates)')
    ax1.legend()
    
    # Plot 2: Angle distributions
    ax2 = fig.add_subplot(132)
    ax2.hist(right_angles, bins=20, alpha=0.7, label='Right Hand', color='red')
    ax2.hist(left_angles, bins=20, alpha=0.7, label='Left Hand', color='blue')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Count')
    ax2.set_title('Hand Joint Angle Distribution')
    ax2.legend()
    
    # Plot 3: Coordinate system reference
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Draw coordinate system axes
    origin = np.array([0, 0, 0])
    axes_length = 10
    
    # X-axis (red)
    ax3.quiver(origin[0], origin[1], origin[2], axes_length, 0, 0, 
              color='red', arrow_length_ratio=0.1, label='X (Lateral)')
    # Y-axis (green)
    ax3.quiver(origin[0], origin[1], origin[2], 0, axes_length, 0, 
              color='green', arrow_length_ratio=0.1, label='Y (Vertical)')
    # Z-axis (blue)
    ax3.quiver(origin[0], origin[1], origin[2], 0, 0, axes_length, 
              color='blue', arrow_length_ratio=0.1, label='Z (Depth)')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Coordinate System\nReference')
    ax3.legend()
    
    plt.tight_layout()
    plt.suptitle(f'PianoMotion10M Coordinate System Analysis - Frame {frame_idx}', y=1.02)
    plt.show()
```

### 5. Integrate Complete Coordinate System Handling

Update the main processing pipeline:

```python
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
```

## Testing the Fix

1. **Scaling validation**: Verify that scaled positions are in reasonable ranges
2. **Coordinate system consistency**: Check that X, Y, Z axes make sense for piano playing
3. **Spatial scale verification**: Compare hand movement ranges with typical piano playing
4. **Visualization accuracy**: Ensure visualizations show realistic hand motions
5. **Comparison with official**: Match coordinate handling with official inference code

## Impact of Fix

- **Correct spatial representation**: Hand positions now properly scaled and positioned
- **Accurate coordinate system**: X, Y, Z axes correctly interpreted
- **Realistic spatial scale**: Hand movements in physically plausible ranges
- **Better visualization**: Coordinate-aware visualization shows meaningful hand motions
- **Piano context**: Spatial relationship to piano keyboard properly understood
- **Compatibility with official code**: Matches coordinate handling in official implementation 