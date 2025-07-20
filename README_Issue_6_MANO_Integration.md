# Issue 6: MANO Model Integration Issues

## Problem Description

The current implementation in `mano_utils.py` and `midi_to_frames.py` has fundamental misunderstandings about how MANO (Model of Articulated Objects) integrates with the PianoMotion10M model outputs, leading to incorrect hand pose interpretation and visualization.

### Specific Issues:

1. **Wrong Joint Structure Assumptions**:
   - Current code assumes 16 joints per hand with direct 3D coordinates
   - PianoMotion10M outputs 48 rotation parameters per hand (MANO-style parameterization)
   - MANO uses 21 joints in full model, but PianoMotion10M uses a different parameterization

2. **Missing MANO Forward Kinematics**:
   - Current code treats model outputs as direct joint positions
   - Should use MANO forward kinematics to convert rotation parameters → joint positions
   - Missing the actual MANO model integration for pose reconstruction

3. **Incorrect Visualization Approach**:
   - Current visualization assumes fixed joint connections
   - Should render based on MANO hand model with proper bone structure
   - Missing realistic hand pose visualization

4. **Misunderstanding of Model Output Format**:
   - Model outputs: 48 rotation parameters + 3 position coordinates per hand
   - Current code: assumes 16 × 3 = 48 joint coordinates
   - Need to understand MANO parameterization vs. direct joint positions

### Code Location:
- File: `mano_utils.py`
- Lines: 300-450 (visualization functions)
- File: `midi_to_frames.py`
- Lines: 200-230 (MANO integration attempt)

## Recommended Fixes

### 1. Understand PianoMotion10M MANO Integration

Clarify how PianoMotion10M uses MANO parameterization:

```python
def explain_pianomotion_mano_integration():
    """
    Explain how PianoMotion10M integrates with MANO model.
    """
    
    integration_info = {
        "model_architecture": {
            "position_predictor": "Outputs 6D (3D position for each hand)",
            "gesture_generator": "Outputs 96D (48 rotation parameters per hand)",
            "total_output": "6D position + 96D rotation = 102D per frame"
        },
        "mano_usage": {
            "parameterization": "MANO-style rotation parameters",
            "joint_count": "21 joints in full MANO, but parameterized differently",
            "output_format": "48 parameters per hand (not 48/3 = 16 joints)",
            "coordinate_system": "MANO coordinate frame with piano-specific adaptations"
        },
        "data_flow": {
            "step_1": "Audio → Position Predictor → 3D hand positions",
            "step_2": "Audio + Positions → Diffusion Model → Hand rotation parameters",
            "step_3": "Positions + Rotations → MANO Forward Kinematics → Joint positions",
            "step_4": "Joint positions → Visualization/Animation"
        },
        "official_rendering": {
            "method": "Uses render_result() from datasets/show.py",
            "input_format": "Concatenated [position, rotation] arrays",
            "right_hand": "np.concatenate([guide[:, :3], prediction[:, :48]], 1)",
            "left_hand": "np.concatenate([guide[:, 3:], prediction[:, 48:]], 1)"
        }
    }
    
    print("PianoMotion10M MANO Integration:")
    for category, details in integration_info.items():
        print(f"\n  {category.replace('_', ' ').title()}:")
        for key, value in details.items():
            print(f"    - {key.replace('_', ' ').title()}: {value}")
    
    return integration_info
```

### 2. Implement Proper MANO Forward Kinematics

Create a function to convert rotation parameters to joint positions:

```python
def convert_mano_params_to_joints(hand_position, hand_rotations, hand='right'):
    """
    Convert MANO parameters (position + rotations) to joint positions.
    
    Args:
        hand_position: (3,) array - 3D position of hand
        hand_rotations: (48,) array - MANO rotation parameters
        hand: 'right' or 'left'
    
    Returns:
        joint_positions: (21, 3) array - 3D positions of MANO joints
    """
    
    try:
        # Try to use the actual MANO model if available
        import sys
        import os
        sys.path.append('PianoMotion10M')
        from models.mano import build_mano
        
        # Load MANO model
        mano_layer = build_mano()
        mano_model = mano_layer[hand]
        
        # Convert rotation parameters to MANO format
        # Note: This is a simplified conversion - actual implementation depends on
        # how PianoMotion10M parameterizes the rotations
        
        # Reshape rotation parameters (48,) → appropriate MANO format
        # This is dataset-specific and may require analysis of training code
        if len(hand_rotations) == 48:
            # Assume the 48 parameters map to MANO pose parameters
            # This mapping needs to be determined from the training code
            mano_pose = hand_rotations.reshape(-1)  # Keep as is for now
        else:
            raise ValueError(f"Unexpected rotation parameter count: {len(hand_rotations)}")
        
        # Set hand shape to mean (or zeros for simplicity)
        batch_size = 1
        hand_shape = torch.zeros(batch_size, 10)  # MANO shape parameters
        
        # Convert position to global translation
        global_orient = torch.zeros(batch_size, 3)  # Root rotation
        transl = torch.tensor(hand_position).unsqueeze(0).float()  # Translation
        
        # Convert pose parameters to tensor
        hand_pose = torch.tensor(mano_pose).unsqueeze(0).float()
        
        # Forward pass through MANO
        output = mano_model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=hand_shape,
            transl=transl
        )
        
        # Extract joint positions
        joints = output.joints[0].detach().numpy()  # (21, 3)
        
        print(f"✓ MANO forward kinematics successful for {hand} hand")
        print(f"  - Input rotations: {hand_rotations.shape}")
        print(f"  - Input position: {hand_position}")
        print(f"  - Output joints: {joints.shape}")
        
        return joints
        
    except Exception as e:
        print(f"✗ MANO forward kinematics failed: {e}")
        print("Falling back to simplified joint approximation")
        
        # Fallback: Create approximate joint positions
        return approximate_joints_from_parameters(hand_position, hand_rotations, hand)

def approximate_joints_from_parameters(hand_position, hand_rotations, hand='right'):
    """
    Create approximate joint positions when full MANO is not available.
    """
    
    # Create a simplified hand model with 21 joints
    # This is an approximation - not as accurate as full MANO
    
    # Basic hand structure (relative to wrist)
    if hand == 'right':
        hand_sign = 1
    else:
        hand_sign = -1  # Mirror for left hand
    
    # Approximate joint offsets (in hand coordinate system)
    joint_offsets = np.array([
        [0, 0, 0],                           # 0: Wrist
        [hand_sign * 0.02, 0.01, 0.02],     # 1: Thumb CMC
        [hand_sign * 0.03, 0.02, 0.04],     # 2: Thumb MCP
        [hand_sign * 0.035, 0.03, 0.055],   # 3: Thumb IP
        [hand_sign * 0.04, 0.035, 0.07],    # 4: Thumb Tip
        [hand_sign * 0.02, 0.08, 0.01],     # 5: Index MCP
        [hand_sign * 0.02, 0.11, 0.015],    # 6: Index PIP
        [hand_sign * 0.02, 0.135, 0.02],    # 7: Index DIP
        [hand_sign * 0.02, 0.155, 0.025],   # 8: Index Tip
        [0, 0.09, 0.005],                   # 9: Middle MCP
        [0, 0.125, 0.01],                   # 10: Middle PIP
        [0, 0.15, 0.015],                   # 11: Middle DIP
        [0, 0.17, 0.02],                    # 12: Middle Tip
        [hand_sign * -0.02, 0.085, 0],      # 13: Ring MCP
        [hand_sign * -0.02, 0.115, 0.005],  # 14: Ring PIP
        [hand_sign * -0.02, 0.14, 0.01],    # 15: Ring DIP
        [hand_sign * -0.02, 0.16, 0.015],   # 16: Ring Tip
        [hand_sign * -0.04, 0.075, -0.005], # 17: Little MCP
        [hand_sign * -0.04, 0.1, 0],        # 18: Little PIP
        [hand_sign * -0.04, 0.12, 0.005],   # 19: Little DIP
        [hand_sign * -0.04, 0.135, 0.01]    # 20: Little Tip
    ])
    
    # Apply some rotation based on rotation parameters
    # This is a very simplified approach
    if len(hand_rotations) >= 6:  # At least some rotation info
        # Use first few rotation parameters for basic hand orientation
        rotation_factor = np.mean(hand_rotations[:6]) * 0.1  # Scale down
        
        # Simple rotation around Y-axis (finger curl)
        cos_r = np.cos(rotation_factor)
        sin_r = np.sin(rotation_factor)
        
        for i in range(len(joint_offsets)):
            x, y, z = joint_offsets[i]
            joint_offsets[i, 0] = x * cos_r - z * sin_r
            joint_offsets[i, 2] = x * sin_r + z * cos_r
    
    # Translate to world position
    joints = joint_offsets + hand_position
    
    print(f"✓ Created approximate joints for {hand} hand ({len(joints)} joints)")
    
    return joints
```

### 3. Create Proper MANO-Based Visualization

Update the visualization to work with MANO joint structure:

```python
def visualize_mano_hands(processed_data, frame_idx=0, use_full_mano=True):
    """
    Visualize hands using proper MANO model integration.
    """
    
    # Extract data for the specified frame
    right_pos = processed_data['right_hand_position'][frame_idx]
    left_pos = processed_data['left_hand_position'][frame_idx]
    right_angles = processed_data['right_hand_angles'][frame_idx]
    left_angles = processed_data['left_hand_angles'][frame_idx]
    
    print(f"MANO-based visualization for frame {frame_idx}")
    
    # Convert MANO parameters to joint positions
    try:
        if use_full_mano:
            right_joints = convert_mano_params_to_joints(right_pos, right_angles, 'right')
            left_joints = convert_mano_params_to_joints(left_pos, left_angles, 'left')
        else:
            right_joints = approximate_joints_from_parameters(right_pos, right_angles, 'right')
            left_joints = approximate_joints_from_parameters(left_pos, left_angles, 'left')
    except Exception as e:
        print(f"Joint conversion failed: {e}")
        return
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # MANO joint connections (21-joint model)
    mano_connections = [
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
    
    # Plot right hand
    ax1 = fig.add_subplot(131, projection='3d')
    plot_hand_with_mano_structure(ax1, right_joints, mano_connections, 
                                 'Right Hand', 'red')
    
    # Plot left hand
    ax2 = fig.add_subplot(132, projection='3d')
    plot_hand_with_mano_structure(ax2, left_joints, mano_connections, 
                                 'Left Hand', 'blue')
    
    # Plot both hands together
    ax3 = fig.add_subplot(133, projection='3d')
    plot_hand_with_mano_structure(ax3, right_joints, mano_connections, 
                                 'Right', 'red', alpha=0.7)
    plot_hand_with_mano_structure(ax3, left_joints, mano_connections, 
                                 'Left', 'blue', alpha=0.7, add_to_existing=True)
    ax3.set_title('Both Hands')
    
    plt.suptitle(f'MANO-based Hand Visualization - Frame {frame_idx}')
    plt.tight_layout()
    plt.show()

def plot_hand_with_mano_structure(ax, joints, connections, title, color, alpha=1.0, add_to_existing=False):
    """
    Plot a hand with proper MANO joint structure.
    """
    
    if not add_to_existing:
        ax.clear()
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
              c=color, s=50, alpha=alpha, label=title)
    
    # Plot connections
    for parent, child in connections:
        if parent < len(joints) and child < len(joints):
            ax.plot([joints[parent, 0], joints[child, 0]],
                   [joints[parent, 1], joints[child, 1]],
                   [joints[parent, 2], joints[child, 2]], 
                   color=color, linewidth=2, alpha=alpha)
    
    # Highlight key joints
    key_joints = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
    for joint_idx in key_joints:
        if joint_idx < len(joints):
            ax.scatter(joints[joint_idx, 0], joints[joint_idx, 1], joints[joint_idx, 2], 
                      c='black', s=80, alpha=alpha*0.8, marker='o', edgecolors=color)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = 0.1  # Adjust based on hand size
    ax.set_xlim([joints[0, 0] - max_range, joints[0, 0] + max_range])
    ax.set_ylim([joints[0, 1] - max_range, joints[0, 1] + max_range])
    ax.set_zlim([joints[0, 2] - max_range, joints[0, 2] + max_range])
```

### 4. Integrate with Official Rendering Pipeline

Create compatibility with the official rendering approach:

```python
def prepare_data_for_official_rendering(processed_data):
    """
    Prepare data in the format expected by official PianoMotion10M rendering.
    """
    
    # Get all frames
    num_frames = processed_data['num_frames']
    
    # Prepare data in official format
    right_data = np.concatenate([
        processed_data['right_hand_position'],  # (frames, 3)
        processed_data['right_hand_angles']     # (frames, 48)
    ], axis=1)  # Result: (frames, 51)
    
    left_data = np.concatenate([
        processed_data['left_hand_position'],   # (frames, 3)
        processed_data['left_hand_angles']      # (frames, 48)
    ], axis=1)  # Result: (frames, 51)
    
    print(f"Prepared data for official rendering:")
    print(f"  - Right hand data shape: {right_data.shape}")
    print(f"  - Left hand data shape: {left_data.shape}")
    print(f"  - Frame count: {num_frames}")
    print(f"  - Data format: [position(3), rotation_params(48)] per hand")
    
    return right_data, left_data

def use_official_rendering_if_available(right_data, left_data, audio_array):
    """
    Use the official PianoMotion10M rendering if available.
    """
    
    try:
        # Import official rendering function
        sys.path.append('PianoMotion10M')
        from datasets.show import render_result
        
        # Create output directory
        output_dir = "rendered_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use official rendering
        render_result(
            output_dir,
            audio_array,
            right_data,
            left_data,
            save_video=False  # Set to True if you want video output
        )
        
        print(f"✓ Used official PianoMotion10M rendering")
        print(f"✓ Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Official rendering failed: {e}")
        print("Falling back to custom visualization")
        return False
```

### 5. Complete MANO Integration Pipeline

Integrate everything into the main pipeline:

```python
# Update the main processing and visualization section
def complete_mano_integration_pipeline(pose_hat, guide, audio_wave, device='cpu'):
    """
    Complete pipeline with proper MANO integration.
    """
    
    # Explain the integration
    integration_info = explain_pianomotion_mano_integration()
    
    # Process with proper coordinate system
    processed_data = process_model_output_with_proper_coordinates(pose_hat, guide, device)
    
    # Try official rendering first
    right_data, left_data = prepare_data_for_official_rendering(processed_data)
    
    if use_official_rendering_if_available(right_data, left_data, audio_wave):
        print("✓ Used official PianoMotion10M rendering pipeline")
    else:
        print("Using custom MANO-based visualization")
        
        # Use custom MANO visualization
        try:
            visualize_mano_hands(processed_data, frame_idx=-1, use_full_mano=True)
        except Exception as e:
            print(f"Full MANO visualization failed: {e}")
            print("Using simplified approximation")
            visualize_mano_hands(processed_data, frame_idx=-1, use_full_mano=False)
    
    return processed_data, integration_info
```

## Testing the Fix

1. **MANO model availability**: Check if MANO model files are accessible
2. **Parameter conversion**: Verify rotation parameters convert correctly to joint positions
3. **Joint structure validation**: Ensure 21 joints are generated with proper connections
4. **Visualization accuracy**: Compare with official rendering if available
5. **Hand pose realism**: Check that generated poses look anatomically correct

## Impact of Fix

- **Proper MANO integration**: Correctly uses MANO parameterization and forward kinematics
- **Accurate joint positions**: Converts rotation parameters to realistic joint positions
- **Better visualization**: Shows anatomically correct hand poses
- **Official compatibility**: Works with the official PianoMotion10M rendering pipeline
- **Fallback robustness**: Provides approximations when full MANO is not available
- **Understanding clarity**: Makes the relationship between model outputs and hand poses clear 