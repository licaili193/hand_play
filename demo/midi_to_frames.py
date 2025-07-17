import argparse
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_hand_model():
    """Load hand model from JSON file. Program fails if not found."""
    try:
        with open("hand_model.json", "r") as f:
            model_data = json.load(f)
        hand_points = np.array(model_data["hand_points"])
        hand_bones = model_data["hand_bones"]
        print("✓ Loaded hand model from hand_model.json")
        return hand_points, hand_bones
    except FileNotFoundError:
        print("❌ Error: hand_model.json not found!")
        print("Please run the setup script to download the hand model:")
        print("  python setup_pianomotion.ps1")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in hand_model.json: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: Missing required key in hand_model.json: {e}")
        sys.exit(1)

# Load hand model
HAND_POINTS, HAND_BONES = load_hand_model()

try:
    from PianoMotion10M.datasets import utils as pm_utils
    print("✓ Using installed PianoMotion10M package")
except ImportError:  # Library not installed, try local clone
    import sys
    import os
    
    # Check if we have a local clone
    local_repo = "PianoMotion10M"
    if os.path.exists(local_repo):
        sys.path.append(local_repo)
        try:
            from datasets import utils as pm_utils
            print(f"✓ Using local PianoMotion10M repository at '{local_repo}'")
        except ImportError:
            print(f"Error: PianoMotion10M repository found at '{local_repo}' but import failed.")
            print("Please run the setup script: .\\setup_pianomotion.ps1")
            sys.exit(1)
    else:
        print("Error: PianoMotion10M repository not found.")
        print("Please run the setup script: .\\setup_pianomotion.ps1")
        sys.exit(1)

print("✓ Using 2D skeleton rendering")

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

def midi_to_frames(midi_path, output_dir, fps=30):
    print(f"Processing MIDI file: {midi_path}")
    events, times = read_midi_general(midi_path)
    duration = times[-1] if len(times) > 0 else 0
    print(f"MIDI duration: {duration:.2f} seconds")
    
    print("Computing frame roll...")
    frame_roll = read_frame_roll(times, events, duration, fps)
    print(f"Generated {len(frame_roll)} frames at {fps} FPS")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("Generating 2D skeleton frames...")
    for i, frame in enumerate(frame_roll):
        notes = np.where(frame > 0)[0]
        left = [n for n in notes if n < 44]
        right = [n for n in notes if n >= 44]
        lx = np.mean([note_to_x(n) for n in left]) if left else -0.5
        rx = np.mean([note_to_x(n) for n in right]) if right else 0.5

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.6, 0.8)
        ax.axis('off')

        def draw_hand(points, color):
            for a, b in HAND_BONES:
                ax.plot([points[a,0], points[b,0]],
                        [points[a,1], points[b,1]],
                        color=color, linewidth=2)

        right_hand = HAND_POINTS.copy()
        right_hand[:,0] += rx
        draw_hand(right_hand, 'red')

        left_hand = HAND_POINTS.copy()
        left_hand[:,0] = -left_hand[:,0]  # mirror
        left_hand[:,0] += lx
        draw_hand(left_hand, 'blue')

        fig.savefig(os.path.join(output_dir, f"frame_{i:04d}.png"))
        plt.close(fig)
    
    print(f"✓ Successfully generated {len(frame_roll)} frames!")

def main():
    parser = argparse.ArgumentParser(description="Generate simple hand position frames from MIDI")
    parser.add_argument('midi', help='Path to MIDI file')
    parser.add_argument('--out', default='demo_frames', help='Output directory')
    args = parser.parse_args()
    midi_to_frames(args.midi, args.out)

if __name__ == '__main__':
    main()
