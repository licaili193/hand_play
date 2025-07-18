import argparse
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add video generation imports
try:
    import imageio.v2 as imageio  # Use v2 to avoid deprecation warning
    from moviepy import VideoFileClip, AudioFileClip
    import soundfile as sf
    import librosa
    import librosa.display
    AUDIO_AVAILABLE = True
    VIDEO_AVAILABLE = True
    print("✓ Video and audio generation libraries available")
except ImportError as e:
    VIDEO_AVAILABLE = False
    AUDIO_AVAILABLE = False
    print(f"⚠️  Video/audio generation libraries not available: {e}")
    print("Install with: pip install imageio moviepy soundfile librosa")

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

def analyze_hand_usage(frame_roll):
    """Analyze MIDI to determine if it's single-hand or two-hand piece."""
    print("Analyzing hand usage patterns...")
    
    # Collect all notes and their frequencies
    note_frequencies = defaultdict(int)
    frame_hand_usage = []
    
    for frame_idx, frame in enumerate(frame_roll):
        notes = np.where(frame > 0)[0]
        if len(notes) == 0:
            frame_hand_usage.append({'type': 'none', 'notes': []})
            continue
            
        # Count note frequencies
        for note in notes:
            note_frequencies[note] += 1
        
        # Analyze note range for this frame
        note_range = max(notes) - min(notes) if len(notes) > 1 else 0
        note_span = len(set(notes))
        
        # Determine if this frame suggests single-hand or two-hand
        if note_range > 20 or note_span > 6:  # Large range or many notes = likely two hands
            frame_hand_usage.append({'type': 'two_hands', 'notes': notes})
        elif note_range < 10 and note_span <= 3:  # Small range, few notes = likely single hand
            frame_hand_usage.append({'type': 'single_hand', 'notes': notes})
        else:
            frame_hand_usage.append({'type': 'uncertain', 'notes': notes})
    
    # Analyze overall piece characteristics
    all_notes = list(note_frequencies.keys())
    if len(all_notes) == 0:
        return 'single_hand', frame_hand_usage
    
    overall_range = max(all_notes) - min(all_notes)
    total_unique_notes = len(all_notes)
    
    # Determine piece type based on overall characteristics
    if overall_range > 30 or total_unique_notes > 15:
        piece_type = 'two_hands'
        print(f"✓ Detected: Two-hand piece (range: {overall_range}, unique notes: {total_unique_notes})")
    elif overall_range < 15 and total_unique_notes < 8:
        piece_type = 'single_hand'
        print(f"✓ Detected: Single-hand piece (range: {overall_range}, unique notes: {total_unique_notes})")
    else:
        piece_type = 'two_hands'  # Default to two hands for safety
        print(f"⚠️  Uncertain: Defaulting to two-hand piece (range: {overall_range}, unique notes: {total_unique_notes})")
    
    return piece_type, frame_hand_usage

def smart_hand_assignment(notes, piece_type, frame_hand_usage, frame_idx):
    """Intelligently assign notes to hands based on piece type and context."""
    
    if len(notes) == 0:
        return [], []
    
    if piece_type == 'single_hand':
        # For single-hand pieces, assign all notes to one hand
        # Determine which hand based on note range (lower notes = left hand, higher = right hand)
        avg_note = np.mean(notes)
        if avg_note < 44:  # Middle C is around note 60, so <44 is lower register
            return notes, []  # Left hand
        else:
            return [], notes  # Right hand
    
    elif piece_type == 'two_hands':
        # For two-hand pieces, use smarter assignment
        if len(notes) <= 2:
            # Few notes - assign based on position
            left_notes = [n for n in notes if n < 44]
            right_notes = [n for n in notes if n >= 44]
            return left_notes, right_notes
        
        else:
            # Many notes - use clustering to separate hands
            notes_array = np.array(notes)
            
            # Simple clustering: split at the median
            median_note = np.median(notes_array)
            left_notes = [n for n in notes if n < median_note]
            right_notes = [n for n in notes if n >= median_note]
            
            # Ensure both hands have notes if possible
            if len(left_notes) == 0 and len(right_notes) > 1:
                # Move one note to left hand
                right_notes.sort()
                left_notes = [right_notes.pop(0)]
            elif len(right_notes) == 0 and len(left_notes) > 1:
                # Move one note to right hand
                left_notes.sort()
                right_notes = [left_notes.pop(-1)]
            
            return left_notes, right_notes
    
    else:
        # Fallback to simple split
        left_notes = [n for n in notes if n < 44]
        right_notes = [n for n in notes if n >= 44]
        return left_notes, right_notes

def generate_smart_hand_poses(frame_roll, fps=30):
    """Generate hand poses with intelligent hand assignment."""
    print("Generating smart hand poses...")
    
    # Analyze the piece first
    piece_type, frame_hand_usage = analyze_hand_usage(frame_roll)
    
    poses = []
    num_frames = len(frame_roll)
    
    for frame_idx in range(num_frames):
        frame = frame_roll[frame_idx]
        notes = np.where(frame > 0)[0]
        
        # Use smart hand assignment
        left_notes, right_notes = smart_hand_assignment(notes, piece_type, frame_hand_usage, frame_idx)
        
        # Generate pose parameters
        pose = generate_pose_from_notes_smart(left_notes, right_notes, frame_idx, fps, piece_type)
        poses.append(pose)
    
    return poses, piece_type

def generate_pose_from_notes_smart(left_notes, right_notes, frame_idx, fps, piece_type):
    """Generate pose parameters with smart positioning to avoid overlap."""
    
    time_factor = frame_idx / fps
    
    # Calculate hand positions
    if piece_type == 'single_hand':
        # Single hand - center it
        if left_notes:
            left_x = np.mean([note_to_x(n) for n in left_notes])
            right_x = 0.8  # Move right hand out of the way
        elif right_notes:
            right_x = np.mean([note_to_x(n) for n in right_notes])
            left_x = -0.8  # Move left hand out of the way
        else:
            left_x, right_x = -0.5, 0.5
    else:
        # Two hands - position based on notes
        left_x = np.mean([note_to_x(n) for n in left_notes]) if left_notes else -0.5
        right_x = np.mean([note_to_x(n) for n in right_notes]) if right_notes else 0.5
    
    # Generate finger movements
    left_finger_angles = generate_finger_angles(left_notes, frame_idx, fps, 'left')
    right_finger_angles = generate_finger_angles(right_notes, frame_idx, fps, 'right')
    
    # Hand height variation
    left_height = 0.0 + 0.1 * np.sin(time_factor * 2)
    right_height = 0.0 + 0.1 * np.sin(time_factor * 2 + np.pi)
    
    # Add movement when notes are active
    if left_notes:
        left_height += 0.05 * len(left_notes) / 10
    if right_notes:
        right_height += 0.05 * len(right_notes) / 10
    
    # For single-hand pieces, adjust inactive hand
    if piece_type == 'single_hand':
        if not left_notes:
            left_height = -0.3  # Move inactive hand down
        if not right_notes:
            right_height = -0.3  # Move inactive hand down
    
    pose = {
        'left_hand': {
            'position': [left_x, left_height, 0.0],
            'angles': left_finger_angles,
            'notes': left_notes,
            'active': len(left_notes) > 0
        },
        'right_hand': {
            'position': [right_x, right_height, 0.0],
            'angles': right_finger_angles,
            'notes': right_notes,
            'active': len(right_notes) > 0
        },
        'piece_type': piece_type
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

def midi_to_audio(midi_path, output_path, sr=22050):
    """Convert MIDI file to audio using mido and numpy."""
    if not AUDIO_AVAILABLE:
        print("❌ Audio generation libraries not available. Skipping audio creation.")
        return False
    
    try:
        print(f"Converting MIDI to audio: {midi_path}")
        
        # Load MIDI file
        midi_file = MidiFile(midi_path)
        
        # Get tempo (default to 120 BPM if not found)
        tempo = 500000  # microseconds per beat (120 BPM)
        for msg in midi_file.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        
        # Calculate timing
        ticks_per_beat = midi_file.ticks_per_beat
        beats_per_second = 1e6 / tempo
        ticks_per_second = ticks_per_beat * beats_per_second
        
        # Initialize audio array
        duration_seconds = 0
        for track in midi_file.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
            duration_seconds = max(duration_seconds, track_time / ticks_per_second)
        
        # Create audio array
        audio_length = int(duration_seconds * sr)
        audio = np.zeros(audio_length)
        
        # Process MIDI events to generate audio
        current_time = 0
        active_notes = {}  # {note: (start_time, velocity)}
        
        for track in midi_file.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
                current_time = track_time / ticks_per_second
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note start
                    note = msg.note
                    start_sample = int(current_time * sr)
                    active_notes[note] = (start_sample, msg.velocity)
                    
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note end
                    note = msg.note
                    if note in active_notes:
                        start_sample, velocity = active_notes[note]
                        end_sample = int(current_time * sr)
                        
                        # Generate simple sine wave for the note
                        frequency = 440 * (2 ** ((note - 69) / 12))  # A4 = 440Hz, note 69
                        duration_samples = end_sample - start_sample
                        
                        if duration_samples > 0:
                            t = np.linspace(0, duration_samples / sr, duration_samples)
                            note_audio = np.sin(2 * np.pi * frequency * t) * (velocity / 127.0) * 0.3
                            
                            # Apply simple envelope
                            envelope = np.exp(-t * 2)  # Decay
                            note_audio *= envelope
                            
                            # Add to audio array
                            if start_sample + duration_samples <= len(audio):
                                audio[start_sample:start_sample + duration_samples] += note_audio
                        
                        del active_notes[note]
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save as WAV file
        sf.write(output_path, audio, sr)
        
        print(f"✓ Audio created successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting MIDI to audio: {e}")
        return False

def create_video_from_frames(output_dir, fps=30, video_name="hand_animation.mp4", midi_path=None):
    """Create a video from the generated frame images with optional audio."""
    if not VIDEO_AVAILABLE:
        print("❌ Video generation libraries not available. Skipping video creation.")
        return False
    
    print(f"Creating video from frames...")
    
    # Look for frames in the frames subdirectory
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        print("❌ Frames directory not found.")
        return False
    
    # Get all frame files
    frame_files = []
    for file in os.listdir(frames_dir):
        if file.startswith("smart_frame_") and file.endswith(".png"):
            frame_files.append(file)
    
    if not frame_files:
        print("❌ No frame files found for video creation.")
        return False
    
    # Sort frame files by frame number
    frame_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    print(f"Found {len(frame_files)} frames to combine into video...")
    
    # Create video writer
    video_path = os.path.join(output_dir, video_name)
    temp_video_path = os.path.join(output_dir, "temp_video.mp4")
    temp_audio_path = os.path.join(output_dir, "temp_audio.wav")
    
    try:
        # Create video without audio first
        with imageio.get_writer(temp_video_path, fps=fps) as writer:
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
        
        # Convert to final format with better compression
        video = VideoFileClip(temp_video_path)
        
        # Add audio if MIDI path is provided
        if midi_path and AUDIO_AVAILABLE:
            print("Adding audio to video...")
            if midi_to_audio(midi_path, temp_audio_path):
                audio = AudioFileClip(temp_audio_path)
                video = video.with_audio(audio)
        
        video.write_videofile(video_path, codec='libx264')
        video.close()
        
        # Clean up temporary files (with delay to ensure files are closed)
        import time
        time.sleep(0.5)  # Give time for files to be released
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        print(f"✓ Video created successfully: {video_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        # Clean up temporary files if they exist
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return False

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
            if i in [1, 2, 3, 4]:  # Thumb joints
                transformed_points[i, 0] += np.cos(angle) * 0.03
                transformed_points[i, 1] += np.sin(angle) * 0.03
            else:  # Other fingers
                transformed_points[i, 1] += np.sin(angle) * 0.04
                transformed_points[i, 0] += np.cos(angle) * 0.02
    
    return transformed_points

def midi_to_hands_smart(midi_path, output_dir, fps=30, skip_video=False, video_name="hand_animation.mp4", include_audio=True):
    print(f"Processing MIDI file: {midi_path}")
    events, times = read_midi_general(midi_path)
    duration = times[-1] if len(times) > 0 else 0
    print(f"MIDI duration: {duration:.2f} seconds")
    
    print("Computing frame roll...")
    frame_roll = read_frame_roll(times, events, duration, fps)
    print(f"Generated {len(frame_roll)} frames at {fps} FPS")

    # Generate smart hand poses
    poses, piece_type = generate_smart_hand_poses(frame_roll, fps)

    # Delete and recreate output directory if it exists
    if os.path.exists(output_dir):
        import shutil
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create frames subdirectory
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Frames directory: {frames_dir}")

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

    print(f"Generating smart hand pose frames ({piece_type} piece)...")
    for i, pose in enumerate(poses):
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
            
            # Set alpha based on whether hand is active
            alpha = 0.8 if pose_params['active'] else 0.3
            
            # Draw each finger with different color
            for finger_name, bones in finger_bones.items():
                color = finger_colors[finger_name]
                for a, b in bones:
                    ax.plot([posed_points[a, 0], posed_points[b, 0]],
                           [posed_points[a, 1], posed_points[b, 1]],
                           color=color, linewidth=3, alpha=alpha)
            
            # Draw joint points
            for j, point in enumerate(posed_points):
                ax.plot(point[0], point[1], 'ko', markersize=4, alpha=alpha)

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

        # Add piece type and note information
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
        
        # Add piece type indicator
        piece_text = f'Piece Type: {piece_type.replace("_", " ").title()}'
        ax.text(0.02, 0.80, piece_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

        fig.savefig(os.path.join(frames_dir, f"smart_frame_{i:04d}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Successfully generated {len(poses)} smart hand pose frames!")
    print(f"✓ Piece type: {piece_type.replace('_', ' ').title()}")
    
    # Create video from frames
    if not skip_video:
        # Pass MIDI path for audio if requested
        midi_for_audio = midi_path if include_audio else None
        video_success = create_video_from_frames(output_dir, fps, video_name, midi_for_audio)
        if video_success:
            print(f"✓ Video generation completed!")
        else:
            print("⚠️  Video generation failed, but frames were saved successfully.")
    else:
        print("⏭️  Video generation skipped (--no-video flag used)")

def main():
    parser = argparse.ArgumentParser(description="Generate smart hand poses with intelligent hand assignment")
    parser.add_argument('midi', help='Path to MIDI file')
    parser.add_argument('--out', default='output', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    parser.add_argument('--video-name', default='hand_animation.mp4', help='Output video filename')
    parser.add_argument('--no-audio', action='store_true', help='Skip audio generation in video')
    args = parser.parse_args()
    midi_to_hands_smart(args.midi, args.out, args.fps, args.no_video, args.video_name, not args.no_audio)

if __name__ == '__main__':
    main() 