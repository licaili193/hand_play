import argparse
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import savgol_filter

def load_hand_model():
    """Load hand model from JSON file."""
    try:
        with open("hand_model.json", "r") as f:
            model_data = json.load(f)
        hand_points = np.array(model_data["hand_points"])
        hand_bones = model_data["hand_bones"]
        landmark_names = model_data["landmark_names"]
        print("✓ Loaded hand model from hand_model.json")
        return hand_points, hand_bones, landmark_names
    except FileNotFoundError:
        print("❌ Error: hand_model.json not found!")
        print("Please run the setup script to download the hand model:")
        print("  python setup_pianomotion.ps1")
        sys.exit(1)

# Load hand model
HAND_POINTS, HAND_BONES, LANDMARK_NAMES = load_hand_model()

try:
    from PianoMotion10M.datasets import utils as pm_utils
    print("✓ Using PianoMotion10M utils for MIDI processing")
except ImportError:
    import sys
    import os
    
    # Check if we have a local clone
    local_repo = "PianoMotion10M"
    if os.path.exists(local_repo):
        sys.path.append(local_repo)
        try:
            from datasets import utils as pm_utils
            print(f"✓ Using local PianoMotion10M repository at '{local_repo}'")
        except ImportError as e:
            print(f"Error: PianoMotion10M repository found but import failed: {e}")
            print("Please run the setup script: .\\setup_pianomotion.ps1")
            sys.exit(1)
    else:
        print("Error: PianoMotion10M repository not found.")
        print("Please run the setup script: .\\setup_pianomotion.ps1")
        sys.exit(1)

from mido import MidiFile

def read_midi_general(midi_path):
    """Read a MIDI file with a flexible number of tracks."""
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    tempo = 500000  # default 120bpm
    for msg in midi_file.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break

    beats_per_second = 1e6 / tempo
    ticks_per_second = ticks_per_beat * beats_per_second

    track_idx = 1 if len(midi_file.tracks) > 1 else 0
    message_list = []
    ticks = 0
    time_in_second = []
    for message in midi_file.tracks[track_idx]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    return np.array(message_list), np.array(time_in_second)

def read_frame_roll(midi_events_time, midi_events, duration, fps):
    """Use PianoMotion10M TargetProcessor to compute note activation frames."""
    processor = pm_utils.TargetProcessor(
        segment_seconds=duration,
        frames_per_second=fps,
        begin_note=21,
        classes_num=88,
    )
    target, _, _ = processor.process(
        0, midi_events_time, midi_events, segment_seconds=duration
    )
    return target["frame_roll"]

def note_to_x(note_index):
    return -1.0 + 2.0 * (note_index / 87.0)

def create_synthetic_audio_from_midi(midi_path, duration, sample_rate=16000):
    """Create synthetic audio from MIDI for analysis."""
    print("Creating synthetic audio from MIDI...")
    
    # Generate a simple sine wave audio based on MIDI notes
    audio = np.zeros(int(duration * sample_rate))
    
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500000  # default 120bpm
    
    for msg in midi_file.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    
    beats_per_second = 1e6 / tempo
    ticks_per_second = ticks_per_beat * beats_per_second
    
    track_idx = 1 if len(midi_file.tracks) > 1 else 0
    current_time = 0
    
    for message in midi_file.tracks[track_idx]:
        current_time += message.time / ticks_per_second
        
        if hasattr(message, 'note') and hasattr(message, 'type'):
            if message.type == 'note_on' and message.velocity > 0:
                # Convert MIDI note to frequency
                freq = 440 * (2 ** ((message.note - 69) / 12))
                
                # Calculate note duration (simplified)
                note_duration = 0.5  # 0.5 seconds
                
                # Generate sine wave for this note
                start_sample = int(current_time * sample_rate)
                end_sample = min(int((current_time + note_duration) * sample_rate), len(audio))
                
                if start_sample < len(audio):
                    t = np.arange(end_sample - start_sample) / sample_rate
                    note_audio = 0.1 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 2)
                    audio[start_sample:end_sample] += note_audio[:end_sample - start_sample]
    
    return audio

def generate_realistic_hand_poses(frame_roll, fps=30):
    """Generate realistic hand poses with finger movements based on MIDI notes."""
    print("Generating realistic hand poses with finger movements...")
    
    poses = []
    num_frames = len(frame_roll)
    
    for frame_idx in range(num_frames):
        frame = frame_roll[frame_idx]
        notes = np.where(frame > 0)[0]
        
        # Calculate hand positions based on active notes
        left_notes = [n for n in notes if n < 44]
        right_notes = [n for n in notes if n >= 44]
        
        # Generate pose parameters based on note patterns
        pose = generate_pose_from_notes(left_notes, right_notes, frame_idx, fps)
        poses.append(pose)
    
    return poses

def generate_pose_from_notes(left_notes, right_notes, frame_idx, fps):
    """Generate realistic pose parameters based on note patterns."""
    
    # Base hand positions
    left_x = np.mean([note_to_x(n) for n in left_notes]) if left_notes else -0.5
    right_x = np.mean([note_to_x(n) for n in right_notes]) if right_notes else 0.5
    
    # Generate finger movements based on note patterns
    left_finger_angles = generate_finger_angles(left_notes, frame_idx, fps, 'left')
    right_finger_angles = generate_finger_angles(right_notes, frame_idx, fps, 'right')
    
    # Add natural hand movement
    time_factor = frame_idx / fps
    
    # Hand height variation (simulating pressing keys)
    left_height = 0.0 + 0.1 * np.sin(time_factor * 2)  # Gentle up/down movement
    right_height = 0.0 + 0.1 * np.sin(time_factor * 2 + np.pi)  # Opposite phase
    
    # Add more movement when notes are active
    if left_notes:
        left_height += 0.05 * len(left_notes) / 10  # Press down when playing
    if right_notes:
        right_height += 0.05 * len(right_notes) / 10
    
    pose = {
        'left_hand': {
            'position': [left_x, left_height, 0.0],
            'angles': left_finger_angles,
            'notes': left_notes
        },
        'right_hand': {
            'position': [right_x, right_height, 0.0],
            'angles': right_finger_angles,
            'notes': right_notes
        }
    }
    
    return pose

def generate_finger_angles(notes, frame_idx, fps, hand_side):
    """Generate realistic finger joint angles based on note patterns."""
    
    # Initialize finger angles (20 joints per hand)
    angles = np.zeros(20)
    
    # Define finger ranges (which notes each finger typically plays)
    finger_ranges = {
        'thumb': (0, 10),      # Lower notes
        'index': (10, 25),     # Lower-middle notes
        'middle': (25, 40),    # Middle notes
        'ring': (40, 55),      # Upper-middle notes
        'pinky': (55, 87)      # Upper notes
    }
    
    # Map finger indices to joint indices in the hand model
    finger_joints = {
        'thumb': [1, 2, 3, 4],      # Thumb joints
        'index': [5, 6, 7, 8],      # Index finger joints
        'middle': [9, 10, 11, 12],  # Middle finger joints
        'ring': [13, 14, 15, 16],   # Ring finger joints
        'pinky': [17, 18, 19, 20]   # Pinky finger joints
    }
    
    # Generate base movement pattern
    time_factor = frame_idx / fps
    
    for finger_name, (note_min, note_max) in finger_ranges.items():
        finger_notes = [n for n in notes if note_min <= n <= note_max]
        joint_indices = finger_joints[finger_name]
        
        # Base finger movement (natural swaying)
        base_angle = 0.2 * np.sin(time_factor * 3 + hash(finger_name) % 10)
        
        # Add movement when finger is playing notes
        if finger_notes:
            # Press down motion
            press_angle = 0.4 * len(finger_notes) / 5  # More notes = more bend
            base_angle += press_angle
            
            # Add some randomness for realism
            base_angle += np.random.normal(0, 0.1)
        
        # Apply angles to finger joints (progressive bending)
        for i, joint_idx in enumerate(joint_indices):
            if joint_idx < len(angles):
                # Progressive bending: more bend at finger tips
                bend_factor = (i + 1) / len(joint_indices)
                angles[joint_idx] = base_angle * bend_factor
    
    # Add wrist movement
    if len(notes) > 0:
        # Wrist rotates slightly when playing
        wrist_rotation = 0.1 * np.sin(time_factor * 2) * len(notes) / 10
        angles[0] = wrist_rotation  # Wrist joint
    
    return angles

def apply_pose_to_hand_model(base_points, pose_params):
    """Apply pose parameters to the hand model to get actual joint positions."""
    
    # Start with base hand model
    transformed_points = base_points.copy()
    
    # Apply position offset
    transformed_points[:, 0] += pose_params['position'][0]
    transformed_points[:, 1] += pose_params['position'][1]
    transformed_points[:, 2] += pose_params['position'][2]
    
    # Apply finger angle variations
    angles = pose_params['angles']
    
    # Apply rotations to finger joints
    for i, angle in enumerate(angles):
        if i < len(transformed_points):
            # Apply rotation around the joint
            # This is a simplified approach - in reality, you'd use proper 3D rotations
            
            # For thumb (special case - different plane)
            if i in [1, 2, 3, 4]:  # Thumb joints
                transformed_points[i, 0] += np.cos(angle) * 0.03
                transformed_points[i, 1] += np.sin(angle) * 0.03
            else:  # Other fingers
                transformed_points[i, 1] += np.sin(angle) * 0.04
                transformed_points[i, 0] += np.cos(angle) * 0.02
    
    return transformed_points

def midi_to_poses_simple(midi_path, output_dir, fps=30):
    print(f"Processing MIDI file: {midi_path}")
    events, times = read_midi_general(midi_path)
    duration = times[-1] if len(times) > 0 else 0
    print(f"MIDI duration: {duration:.2f} seconds")
    
    print("Computing frame roll...")
    frame_roll = read_frame_roll(times, events, duration, fps)
    print(f"Generated {len(frame_roll)} frames at {fps} FPS")

    # Generate realistic hand poses
    poses = generate_realistic_hand_poses(frame_roll, fps)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Define finger colors for better visualization
    finger_colors = {
        'thumb': '#FF6B6B',    # Red
        'index': '#4ECDC4',    # Teal
        'middle': '#45B7D1',   # Blue
        'ring': '#96CEB4',     # Green
        'pinky': '#FFEAA7'     # Yellow
    }
    
    # Group bones by finger
    finger_bones = {
        'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
        'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
        'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
        'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
        'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
    }

    print("Generating hand pose frames with realistic finger movements...")
    for i, (frame, pose) in enumerate(zip(frame_roll, poses)):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.0, 1.0)
        ax.axis('off')

        def draw_hand_with_pose(base_points, pose_params, hand_side):
            """Draw hand with actual pose parameters."""
            # Apply pose to hand model
            posed_points = apply_pose_to_hand_model(base_points, pose_params)
            
            # Mirror left hand
            if hand_side == 'left':
                posed_points[:, 0] = -posed_points[:, 0]
            
            # Draw each finger with different color
            for finger_name, bones in finger_bones.items():
                color = finger_colors[finger_name]
                for a, b in bones:
                    ax.plot([posed_points[a, 0], posed_points[b, 0]],
                           [posed_points[a, 1], posed_points[b, 1]],
                           color=color, linewidth=3, alpha=0.8)
            
            # Draw joint points
            for j, point in enumerate(posed_points):
                ax.plot(point[0], point[1], 'ko', markersize=4, alpha=0.7)

        # Draw hands with poses
        draw_hand_with_pose(HAND_POINTS, pose['left_hand'], 'left')
        draw_hand_with_pose(HAND_POINTS, pose['right_hand'], 'right')

        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=3, label=finger.capitalize()) 
                          for finger, color in finger_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Add frame info
        ax.text(0.02, 0.98, f'Frame {i:04d}', transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add note information
        total_notes = len(pose['left_hand']['notes']) + len(pose['right_hand']['notes'])
        if total_notes > 0:
            note_text = f'Notes: {total_notes} active'
            ax.text(0.02, 0.92, note_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Show hand positions
            left_pos = pose['left_hand']['position']
            right_pos = pose['right_hand']['position']
            pos_text = f'Left: ({left_pos[0]:.2f}, {left_pos[1]:.2f}) | Right: ({right_pos[0]:.2f}, {right_pos[1]:.2f})'
            ax.text(0.02, 0.86, pos_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        fig.savefig(os.path.join(output_dir, f"pose_frame_{i:04d}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Successfully generated {len(frame_roll)} pose frames with realistic finger movements!")

def main():
    parser = argparse.ArgumentParser(description="Generate realistic hand poses with finger movements from MIDI")
    parser.add_argument('midi', help='Path to MIDI file')
    parser.add_argument('--out', default='output', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    args = parser.parse_args()
    midi_to_poses_simple(args.midi, args.out, args.fps)

if __name__ == '__main__':
    main() 