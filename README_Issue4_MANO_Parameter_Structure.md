# Issue 4: MANO Parameter Structure Assumption

## Problem Description

The current implementation makes assumptions about how the 48 MANO rotation parameters are structured:

```python
# Current code in mano_utils.py
global_orient = hand_rotations[:3]  # First 3 parameters
hand_pose = hand_rotations[3:48]    # Next 45 parameters (15 joints × 3)
```

**Issue**: The code assumes 48 parameters = 3 global orientation + 45 hand pose parameters. However, PianoMotion10M may use a different MANO parameterization where all 48 parameters represent hand pose, or use a different structure entirely.

## Why This Needs Verification

1. **Incorrect Joint Conversion**: Wrong parameter interpretation leads to invalid joint positions
2. **Poor Visualization**: MANO forward kinematics will produce unrealistic hand poses
3. **Model Architecture Mismatch**: The model's output may not match standard MANO format
4. **Finger Movement Artifacts**: Incorrect parameter mapping can cause unnatural finger motions

## MANO Background

Standard MANO model typically uses:
- **Global orientation**: 3 parameters (root rotation)
- **Hand pose**: 45 parameters (15 joints × 3 rotations = 45)
- **Shape parameters**: 10 parameters (hand shape, not used in motion)

However, PianoMotion10M could use variations:
1. **All pose**: 48 hand pose parameters (16 joints × 3)
2. **Modified structure**: Different joint count or parameter organization
3. **PCA reduced**: Compressed representation of hand poses

## Evidence Analysis

From the current codebase:
```python
# The model outputs 48 parameters per hand
assert diffusion_args.bs_dim == 96, "Diffusion model must use bs_dim=96"
# This gives 96/2 = 48 parameters per hand

# But the conversion assumes 3+45 structure:
finger_params = hand_pose.reshape(5, 9)  # 5 fingers, 9 params each
# This works only if hand_pose has exactly 45 parameters
```

## Verification Steps

### Step 1: Examine Official MANO Usage
```python
def examine_official_mano_usage():
    """Examine how PianoMotion10M uses MANO parameterization."""
    
    try:
        import sys
        sys.path.append('PianoMotion10M')
        
        # Check official MANO model
        from models.mano import build_mano
        mano_layer = build_mano()
        mano_model = mano_layer['right']
        
        print("Official MANO model analysis:")
        print(f"  Model type: {type(mano_model)}")
        
        # Check input/output dimensions
        if hasattr(mano_model, 'num_pca_comps'):
            print(f"  PCA components: {mano_model.num_pca_comps}")
        if hasattr(mano_model, 'pose_dim'):
            print(f"  Pose dimension: {mano_model.pose_dim}")
        
        # Check forward pass with test input
        batch_size = 1
        
        # Test with different parameter structures
        test_configs = [
            ("3 global + 45 pose", 3, 45),
            ("0 global + 48 pose", 0, 48),
            ("6 global + 42 pose", 6, 42)
        ]
        
        for name, global_size, pose_size in test_configs:
            try:
                global_orient = torch.zeros(batch_size, global_size)
                hand_pose = torch.zeros(batch_size, pose_size)
                
                # Try forward pass
                output = mano_model(global_orient=global_orient, hand_pose=hand_pose)
                print(f"  ✓ {name}: Forward pass successful")
                print(f"    Output keys: {list(output.keys()) if isinstance(output, dict) else 'tensor'}")
                
            except Exception as e:
                print(f"  ✗ {name}: Failed - {e}")
        
        return mano_model
        
    except Exception as e:
        print(f"Could not examine official MANO usage: {e}")
        return None

def analyze_diffusion_output_structure():
    """Analyze the structure of diffusion model outputs."""
    
    try:
        # Check if there are any comments or docs about parameter structure
        import inspect
        sys.path.append('PianoMotion10M')
        from models.denoise_diffusion import GaussianDiffusion1D_piano2pose
        
        # Look at the source code for clues
        source = inspect.getsource(GaussianDiffusion1D_piano2pose)
        
        # Search for parameter structure hints
        structure_keywords = ['global_orient', 'hand_pose', '48', 'pose_param', 'mano']
        for keyword in structure_keywords:
            if keyword in source.lower():
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        print(f"Found '{keyword}' in line {i}: {line.strip()}")
        
        return True
        
    except Exception as e:
        print(f"Could not analyze diffusion output structure: {e}")
        return False
```

### Step 2: Test Different Parameter Interpretations
```python
def test_parameter_interpretations(hand_position, hand_rotations):
    """Test different ways to interpret the 48 rotation parameters."""
    
    interpretations = {
        "standard_mano": {
            "global_orient": hand_rotations[:3],
            "hand_pose": hand_rotations[3:48],
            "description": "Standard MANO: 3 global + 45 pose"
        },
        "all_pose": {
            "global_orient": torch.zeros(3),
            "hand_pose": hand_rotations[:48],
            "description": "All pose: 0 global + 48 pose"
        },
        "extended_global": {
            "global_orient": hand_rotations[:6],
            "hand_pose": hand_rotations[6:48],
            "description": "Extended global: 6 global + 42 pose"
        },
        "split_evenly": {
            "global_orient": hand_rotations[:24],
            "hand_pose": hand_rotations[24:48],
            "description": "Even split: 24 + 24"
        }
    }
    
    results = {}
    
    for name, config in interpretations.items():
        try:
            print(f"\nTesting {name}: {config['description']}")
            
            # Try to create joint positions with this interpretation
            joints = convert_mano_params_with_structure(
                hand_position, 
                config['global_orient'], 
                config['hand_pose'],
                name
            )
            
            # Validate the resulting joint positions
            validation_score = validate_joint_positions(joints, name)
            
            results[name] = {
                'joints': joints,
                'validation_score': validation_score,
                'success': True
            }
            
            print(f"  ✓ Success: {name} produced valid joints (score: {validation_score:.2f})")
            
        except Exception as e:
            print(f"  ✗ Failed: {name} - {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Compare results
    successful_interpretations = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_interpretations:
        best_interpretation = max(
            successful_interpretations.items(), 
            key=lambda x: x[1]['validation_score']
        )
        print(f"\n🏆 Best interpretation: {best_interpretation[0]} (score: {best_interpretation[1]['validation_score']:.2f})")
        return best_interpretation[0], results
    else:
        print("\n⚠ No interpretation produced valid results")
        return None, results

def convert_mano_params_with_structure(hand_position, global_orient, hand_pose, method_name):
    """Convert MANO parameters to joint positions with specified structure."""
    
    print(f"  Converting with {method_name}:")
    print(f"    Global orient shape: {global_orient.shape if hasattr(global_orient, 'shape') else len(global_orient)}")
    print(f"    Hand pose shape: {hand_pose.shape if hasattr(hand_pose, 'shape') else len(hand_pose)}")
    
    # Use the appropriate conversion method based on parameter structure
    if len(global_orient) == 3 and len(hand_pose) == 45:
        # Standard MANO structure
        return convert_with_standard_mano(hand_position, global_orient, hand_pose)
    elif len(global_orient) == 0 and len(hand_pose) == 48:
        # All pose parameters
        return convert_with_all_pose(hand_position, hand_pose)
    else:
        # Custom structure - use approximation
        return convert_with_custom_structure(hand_position, global_orient, hand_pose)

def validate_joint_positions(joints, method_name):
    """Validate joint positions and return a quality score."""
    
    if joints is None or len(joints) == 0:
        return 0.0
    
    score = 0.0
    max_score = 6.0
    
    # Check 1: Reasonable number of joints (expecting 21)
    if len(joints) == 21:
        score += 1.0
        print(f"    ✓ Correct joint count: 21")
    else:
        print(f"    ⚠ Unexpected joint count: {len(joints)}")
    
    # Check 2: Realistic joint positions (not NaN, not extreme values)
    if not np.any(np.isnan(joints)) and not np.any(np.isinf(joints)):
        score += 1.0
        print(f"    ✓ No NaN/Inf values")
    else:
        print(f"    ⚠ Contains NaN/Inf values")
    
    # Check 3: Reasonable coordinate ranges
    joint_range = np.ptp(joints, axis=0)  # Range in each dimension
    if all(0.01 <= r <= 0.5 for r in joint_range):  # 1cm to 50cm range
        score += 1.0
        print(f"    ✓ Reasonable coordinate ranges: {joint_range}")
    else:
        print(f"    ⚠ Extreme coordinate ranges: {joint_range}")
    
    # Check 4: Hand structure (fingers extend from wrist)
    wrist_pos = joints[0]
    finger_distances = [np.linalg.norm(joints[i] - wrist_pos) for i in [4, 8, 12, 16, 20]]  # Fingertips
    if all(0.05 <= d <= 0.15 for d in finger_distances):  # 5cm to 15cm from wrist
        score += 1.0
        print(f"    ✓ Realistic finger lengths: {finger_distances}")
    else:
        print(f"    ⚠ Unrealistic finger lengths: {finger_distances}")
    
    # Check 5: Bone length consistency
    bone_lengths = calculate_bone_lengths_simple(joints)
    if bone_lengths and all(0.01 <= bl <= 0.08 for bl in bone_lengths):  # 1cm to 8cm bones
        score += 1.0
        print(f"    ✓ Realistic bone lengths")
    else:
        print(f"    ⚠ Unrealistic bone lengths")
    
    # Check 6: Hand chirality (right hand should have thumb on correct side)
    thumb_direction = joints[4] - joints[0]  # Thumb tip - wrist
    if thumb_direction[0] > 0:  # Assuming right hand, thumb should be positive X
        score += 1.0
        print(f"    ✓ Correct hand chirality")
    else:
        print(f"    ⚠ Incorrect hand chirality")
    
    final_score = score / max_score
    print(f"    Total score: {score}/{max_score} = {final_score:.2f}")
    
    return final_score

def calculate_bone_lengths_simple(joints):
    """Calculate simple bone lengths for validation."""
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)   # Little
    ]
    
    bone_lengths = []
    for parent, child in connections:
        if parent < len(joints) and child < len(joints):
            length = np.linalg.norm(joints[child] - joints[parent])
            bone_lengths.append(length)
    
    return bone_lengths
```

### Step 3: Compare with Official Rendering Input
```python
def analyze_official_rendering_input():
    """Analyze how official rendering expects MANO parameters."""
    
    try:
        # Look at the official rendering function
        sys.path.append('PianoMotion10M')
        from datasets.show import render_result
        
        # Check the source code to see how it processes the input
        import inspect
        source = inspect.getsource(render_result)
        
        print("Analyzing official rendering input format:")
        
        # Look for MANO-related parameter usage
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['mano', 'global', 'pose', '48', '3']):
                print(f"Line {i}: {line.strip()}")
        
        # The function signature suggests:
        # render_result(output_dir, audio_array, right_data, left_data, save_video=False)
        # where right_data = np.concatenate([guide[:, :3], prediction[:, :48]], 1)
        # This means it expects [position(3) + rotation_params(48)] per hand
        
        print("\nOfficial rendering expects:")
        print("  Right hand: [position(3), rotation_params(48)]")
        print("  Left hand: [position(3), rotation_params(48)]")
        print("  This suggests the 48 parameters are used as-is for MANO")
        
        return True
        
    except Exception as e:
        print(f"Could not analyze official rendering: {e}")
        return False
```

## Recommended Fix

Create a robust parameter structure detection and conversion system:

```python
def determine_mano_parameter_structure(sample_rotations):
    """Determine the correct MANO parameter structure through testing."""
    
    print("Determining MANO parameter structure...")
    
    # Test different structures with sample data
    test_position = np.array([0, 0.1, 0.2])  # Sample hand position
    
    # Test each interpretation
    interpretations = test_parameter_interpretations(test_position, sample_rotations)
    best_interpretation, results = interpretations
    
    if best_interpretation:
        print(f"✓ Detected parameter structure: {best_interpretation}")
        return best_interpretation
    else:
        print("⚠ Could not determine parameter structure, using fallback")
        return "standard_mano"  # Conservative fallback

def convert_mano_params_robust(hand_position, hand_rotations, hand='right', structure=None):
    """Convert MANO parameters with robust structure detection."""
    
    if structure is None:
        structure = determine_mano_parameter_structure(hand_rotations)
    
    print(f"Converting MANO parameters using {structure} structure")
    
    try:
        if structure == "standard_mano":
            return convert_with_standard_mano(hand_position, hand_rotations[:3], hand_rotations[3:], hand)
        elif structure == "all_pose":
            return convert_with_all_pose(hand_position, hand_rotations, hand)
        else:
            # Use the best available method
            return convert_with_adaptive_structure(hand_position, hand_rotations, hand)
            
    except Exception as e:
        print(f"MANO conversion failed with {structure}: {e}")
        print("Falling back to approximation method")
        return approximate_joints_from_parameters(hand_position, hand_rotations, hand)

def convert_with_standard_mano(hand_position, global_orient, hand_pose, hand='right'):
    """Convert using standard MANO structure (3 global + 45 pose)."""
    
    # Implementation for standard MANO structure
    # This uses the existing logic in mano_utils.py
    return convert_mano_params_to_joints(hand_position, 
                                       np.concatenate([global_orient, hand_pose]), 
                                       hand)

def convert_with_all_pose(hand_position, hand_rotations, hand='right'):
    """Convert using all-pose structure (48 pose parameters)."""
    
    # When all parameters are pose parameters, there's no global orientation
    # This means the hand orientation is encoded in the pose parameters
    
    # Use a modified version of the conversion
    global_orient = np.zeros(3)  # No global orientation
    hand_pose = hand_rotations    # All 48 are pose parameters
    
    # Use different joint mapping for 48 pose parameters
    return convert_with_extended_pose_params(hand_position, hand_pose, hand)

def convert_with_extended_pose_params(hand_position, pose_params, hand='right'):
    """Convert using extended pose parameters (48 instead of 45)."""
    
    # This suggests 16 joints × 3 parameters = 48
    # Instead of the standard 15 joints × 3 parameters = 45
    
    if len(pose_params) != 48:
        raise ValueError(f"Expected 48 pose parameters, got {len(pose_params)}")
    
    # Reshape to 16 joints × 3 parameters
    joint_rotations = pose_params.reshape(16, 3)
    
    # Create extended joint structure with 16+1=17 joints (including wrist)
    # Map to the standard 21-joint MANO structure
    
    joints = create_joints_from_extended_params(hand_position, joint_rotations, hand)
    
    return joints

def create_joints_from_extended_params(hand_position, joint_rotations, hand='right'):
    """Create joint positions from 16 joint rotation parameters."""
    
    # This is a more sophisticated approach that uses the rotation parameters
    # to compute joint positions through forward kinematics
    
    # For now, use a simplified approach
    # In practice, this would need proper MANO forward kinematics
    
    hand_sign = 1 if hand == 'right' else -1
    
    # Base joint structure (similar to existing approach)
    joints = np.zeros((21, 3))
    joints[0] = hand_position  # Wrist
    
    # Apply rotations to compute finger positions
    # This is a simplified implementation - full MANO would be more complex
    
    finger_base_offsets = [
        [hand_sign * 0.02, 0.01, 0.02],    # Thumb base
        [hand_sign * 0.02, 0.08, 0.01],    # Index base
        [0, 0.09, 0.005],                  # Middle base
        [hand_sign * -0.02, 0.085, 0],     # Ring base
        [hand_sign * -0.04, 0.075, -0.005] # Little base
    ]
    
    finger_joint_indices = [
        [1, 2, 3, 4],      # Thumb
        [5, 6, 7, 8],      # Index
        [9, 10, 11, 12],   # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20]   # Little
    ]
    
    # For each finger, use the rotation parameters to compute joint positions
    for finger_idx, (base_offset, joint_indices) in enumerate(zip(finger_base_offsets, finger_joint_indices)):
        if finger_idx < len(joint_rotations):
            # Use rotation parameters for this finger
            finger_rots = joint_rotations[finger_idx * 3:(finger_idx + 1) * 3]
            
            # Compute finger joints with rotation influence
            for i, joint_idx in enumerate(joint_indices):
                if i == 0:
                    # Base joint
                    joints[joint_idx] = hand_position + base_offset
                else:
                    # Subsequent joints influenced by rotations
                    prev_joint = joints[joint_indices[i-1]]
                    
                    # Simple rotation influence (simplified)
                    if i-1 < len(finger_rots):
                        rotation_factor = finger_rots[i-1] * 0.05
                        offset = np.array([0, 0.025, 0]) * (1 + rotation_factor)
                        joints[joint_idx] = prev_joint + offset
    
    return joints
```

## Integration Instructions

1. **Replace the existing MANO conversion** in `mano_utils.py`:
   ```python
   # Replace convert_mano_params_to_joints with:
   def convert_mano_params_to_joints(hand_position, hand_rotations, hand='right'):
       return convert_mano_params_robust(hand_position, hand_rotations, hand)
   ```

2. **Add structure detection** at startup:
   ```python
   # In main() of midi_to_frames.py, after loading the model:
   print("Detecting MANO parameter structure...")
   sample_rotations = np.random.randn(48) * 0.1  # Small random test
   detected_structure = determine_mano_parameter_structure(sample_rotations)
   ```

3. **Add metadata tracking**:
   ```python
   # In save_output_to_json:
   metadata["mano_parameter_structure"] = detected_structure
   metadata["mano_conversion_method"] = "robust_detection"
   ```

## Testing Strategy

1. **Parameter structure detection** with synthetic data
2. **Joint position validation** with multiple test cases
3. **Visual verification** of resulting hand poses
4. **Comparison** with official MANO model outputs (if available)

## Expected Outcomes

After verification, we expect to find:
- The correct parameter structure used by PianoMotion10M
- Whether it follows standard MANO (3+45) or uses a variant
- The proper way to interpret the 48 rotation parameters
- Improved joint position accuracy and hand pose realism 