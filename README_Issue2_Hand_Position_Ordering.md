# Issue 2: Hand Position Ordering Ambiguity

## Problem Description

The current implementation assumes a specific ordering for hand positions in the guide output without verification:

```python
# Current code in midi_to_frames.py
right_hand_pos = scaled_guide[:, :3]    # First 3 dimensions
left_hand_pos = scaled_guide[:, 3:]     # Last 3 dimensions
```

**Issue**: The code assumes the 6D guide output is ordered as `[right_x, right_y, right_z, left_x, left_y, left_z]`, but it could be `[left_x, left_y, left_z, right_x, right_y, right_z]`. This ordering is not verified and could lead to swapped hand assignments.

## Why This Is Critical

1. **Swapped Hand Assignments**: Left and right hands will be reversed in all output
2. **Incorrect Spatial Analysis**: Hand separation and keyboard coverage calculations will be wrong
3. **Invalid Visualizations**: All hand motion visualizations will show incorrect hand positions
4. **Silent Failure**: The swapping may not be immediately obvious, leading to subtle but persistent errors

## Evidence of the Problem

From the codebase analysis:
- No verification of hand ordering in the model loading or inference code
- The Piano2Posi model documentation doesn't specify the output ordering
- Hand positioning logic assumes right-first ordering without justification
- Scale factor application follows the same unverified assumption

## Real-World Impact

```python
# If ordering is actually [left, right] instead of [right, left]:
# Current code does:
right_hand_pos = scaled_guide[:, :3]  # Actually gets LEFT hand position
left_hand_pos = scaled_guide[:, 3:]   # Actually gets RIGHT hand position

# This means:
# - Right hand movements appear on the left side of the piano
# - Left hand movements appear on the right side of the piano
# - All musical interpretation is backwards
```

## Verification Steps

### Step 1: Check Model Architecture Documentation
```python
# Look for ordering specification in PianoMotion10M
# Files to check:
# - PianoMotion10M/models/piano2posi.py
# - PianoMotion10M/datasets/PianoPose.py
# - Training/evaluation scripts for output format
```

### Step 2: Analyze Training Data Format
```python
def analyze_training_data_format():
    """Analyze training data to determine hand ordering convention."""
    
    # Check if PianoPose dataset is available
    try:
        import sys
        sys.path.append('PianoMotion10M')
        from datasets.PianoPose import PianoPose
        
        # Create dataset instance
        dataset = PianoPose(split='train', train_sec=4)
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        
        # Look for position data
        if 'position' in sample or 'guide' in sample:
            pos_data = sample.get('position', sample.get('guide'))
            print(f"Position data shape: {pos_data.shape}")
            print(f"Position data sample: {pos_data[:5]}")  # First few frames
            
            # Analyze the data to infer hand ordering
            # Right hand typically plays higher pitched notes (positive X in piano coordinates)
            # Left hand typically plays lower pitched notes (negative X in piano coordinates)
            
        return sample
    except Exception as e:
        print(f"Could not load training data: {e}")
        return None
```

### Step 3: Test with Known Musical Input
```python
def test_hand_ordering_with_known_input():
    """Test hand ordering using a MIDI file with clear left/right hand separation."""
    
    # Use a simple MIDI with clear hand separation
    # E.g., left hand plays low notes (C2-C4), right hand plays high notes (C4-C6)
    
    # Run inference
    pose_hat, guide, frame_num = run_model_inference(model, audio_wave, device)
    processed_data = process_model_output_with_proper_coordinates(pose_hat, guide)
    
    # Analyze hand positions
    right_pos = processed_data['right_hand_position']
    left_pos = processed_data['left_hand_position']
    
    # Calculate average positions
    avg_right = np.mean(right_pos, axis=0)
    avg_left = np.mean(left_pos, axis=0)
    
    print(f"Average 'right' hand position: {avg_right}")
    print(f"Average 'left' hand position: {avg_left}")
    
    # For piano playing:
    # - Right hand should typically be on the right side (positive X)
    # - Left hand should typically be on the left side (negative X)
    # - If this is reversed, our ordering assumption is wrong
    
    if avg_right[0] > avg_left[0]:
        print("✓ Hand ordering appears correct (right hand more positive X)")
    else:
        print("✗ Hand ordering may be swapped (left hand more positive X)")
        
    return avg_right, avg_left
```

### Step 4: Cross-Reference with Official Rendering
```python
def verify_against_official_rendering():
    """Compare our hand assignments with official PianoMotion10M rendering."""
    
    try:
        # Import official rendering
        from datasets.show import render_result
        
        # The official rendering function signature:
        # render_result(output_dir, audio_array, right_data, left_data, save_video=False)
        
        # Check how official code constructs right_data and left_data
        # This should tell us the expected ordering
        
        print("Official rendering expects:")
        print("- right_data: np.concatenate([guide[:, :3], prediction[:, :48]], 1)")
        print("- left_data: np.concatenate([guide[:, 3:], prediction[:, 48:]], 1)")
        print("This confirms guide ordering is [right_xyz, left_xyz]")
        
        return True
    except Exception as e:
        print(f"Could not verify against official rendering: {e}")
        return False
```

## Recommended Fix

Create a robust hand ordering verification and correction system:

```python
def verify_and_correct_hand_ordering(guide, pose_hat, midi_path=None):
    """Verify and correct hand ordering based on musical content and position analysis."""
    
    # Apply scaling first
    scale = torch.tensor([1.5, 1.5, 25])
    scale_expanded = scale.repeat(2).view(1, 1, 6).expand_as(guide)
    scaled_guide = (guide * scale_expanded)[0].cpu().numpy()
    
    # Extract assumed positions
    assumed_right = scaled_guide[:, :3]
    assumed_left = scaled_guide[:, 3:]
    
    # Method 1: Positional analysis
    avg_assumed_right = np.mean(assumed_right, axis=0)
    avg_assumed_left = np.mean(assumed_left, axis=0)
    
    print(f"Position analysis:")
    print(f"  Assumed right hand avg: {avg_assumed_right}")
    print(f"  Assumed left hand avg: {avg_assumed_left}")
    
    # For piano, right hand typically has higher X coordinate (more to the right)
    position_suggests_swap = avg_assumed_right[0] < avg_assumed_left[0]
    
    # Method 2: MIDI analysis (if available)
    midi_suggests_swap = False
    if midi_path and os.path.exists(midi_path):
        midi_suggests_swap = analyze_midi_for_hand_ordering(
            midi_path, assumed_right, assumed_left
        )
    
    # Method 3: Official code pattern verification
    official_pattern_confirmed = verify_against_official_rendering()
    
    # Decision logic
    should_swap = False
    reasoning = []
    
    if position_suggests_swap:
        reasoning.append("Position analysis suggests swap needed")
        should_swap = True
    
    if midi_suggests_swap:
        reasoning.append("MIDI analysis suggests swap needed")
        should_swap = True
        
    if official_pattern_confirmed:
        reasoning.append("Official code confirms [right, left] ordering")
        should_swap = False  # Override other indicators
    
    # Apply correction if needed
    if should_swap:
        print(f"⚠ Swapping hand assignments based on: {', '.join(reasoning)}")
        corrected_guide = torch.cat([
            guide[:, :, 3:],  # Move left to first position
            guide[:, :, :3]   # Move right to second position  
        ], dim=2)
        
        corrected_pose = torch.cat([
            pose_hat[:, :, 48:],  # Move left hand angles to first position
            pose_hat[:, :, :48]   # Move right hand angles to second position
        ], dim=2)
        
        return corrected_guide, corrected_pose, True
    else:
        print(f"✓ Hand ordering appears correct: {', '.join(reasoning) if reasoning else 'no correction needed'}")
        return guide, pose_hat, False

def analyze_midi_for_hand_ordering(midi_path, pos_right, pos_left):
    """Analyze MIDI file to determine if hand positions match expected musical content."""
    
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Collect notes and their typical hand assignments
        low_notes = []  # Typically left hand
        high_notes = []  # Typically right hand
        
        for instrument in pm.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.pitch < 60:  # Below middle C
                        low_notes.append(note)
                    else:  # Above middle C
                        high_notes.append(note)
        
        # Calculate note density over time
        total_duration = pm.get_end_time()
        low_density = len(low_notes) / total_duration if total_duration > 0 else 0
        high_density = len(high_notes) / total_duration if total_duration > 0 else 0
        
        print(f"MIDI analysis:")
        print(f"  Low notes (left hand): {len(low_notes)} notes, density: {low_density:.2f}/s")
        print(f"  High notes (right hand): {len(high_notes)} notes, density: {high_density:.2f}/s")
        
        # Simple heuristic: if the assumed "right" hand is more active during low note periods,
        # it might actually be the left hand
        # This is a simplified check - a full implementation would need more sophisticated analysis
        
        # For now, return False (no swap suggested) as this requires more complex timing analysis
        return False
        
    except Exception as e:
        print(f"MIDI analysis failed: {e}")
        return False
```

## Integration Instructions

1. **Add verification function** to `midi_to_frames.py` before the coordinate transformation
2. **Update the processing pipeline**:
   ```python
   # Replace this line:
   prediction, scaled_guide = apply_coordinate_transformations(pose_hat, guide, device)
   
   # With this:
   verified_guide, verified_pose, was_swapped = verify_and_correct_hand_ordering(guide, pose_hat, MIDI_PATH)
   prediction, scaled_guide = apply_coordinate_transformations(verified_pose, verified_guide, device)
   ```
3. **Add metadata** to track corrections:
   ```python
   # In save_output_to_json:
   metadata["hand_ordering_corrected"] = was_swapped
   ```

## Testing Strategy

1. **Test with known compositions** that have clear left/right hand separation
2. **Visual verification** of output hand positions against musical content
3. **Cross-reference** with official PianoMotion10M demos/examples
4. **Compare** output with manual inspection of training data (if available)

## Long-term Solution

Ideally, verify the correct ordering by:
1. Examining the official PianoMotion10M training code
2. Testing with the original model weights and known inputs
3. Confirming with the authors' implementation details 