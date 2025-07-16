import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

try:
    from PianoMotion10M.datasets import utils as pm_utils
except ImportError:  # Library not installed, clone on demand
    import subprocess, sys, tempfile
    repo_url = "https://github.com/agnJason/PianoMotion10M.git"
    tmpdir = tempfile.mkdtemp()
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, tmpdir])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa", "soundfile", "tqdm"], stdout=subprocess.DEVNULL)
    sys.path.append(tmpdir)
    from datasets import utils as pm_utils

try:
    # Rendering utilities (requires GPU and nvdiffrast)
    from PianoMotion10M.datasets.show import render_result_frame
    import torch
    HAS_RENDER = torch.cuda.is_available()
except Exception as e:  # pragma: no cover - optional dependency
    HAS_RENDER = False

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
    events, times = read_midi_general(midi_path)
    duration = times[-1] if len(times) > 0 else 0
    frame_roll = read_frame_roll(times, events, duration, fps)

    os.makedirs(output_dir, exist_ok=True)

    # Prepare poses for optional advanced rendering
    if HAS_RENDER:
        poses_left = np.zeros((len(frame_roll), 51), dtype=np.float32)
        poses_right = np.zeros_like(poses_left)

    for i, frame in enumerate(frame_roll):
        notes = np.where(frame > 0)[0]
        left = [n for n in notes if n < 44]
        right = [n for n in notes if n >= 44]
        lx = np.mean([note_to_x(n) for n in left]) if left else -0.5
        rx = np.mean([note_to_x(n) for n in right]) if right else 0.5

        if HAS_RENDER:
            poses_left[i, 0] = lx
            poses_left[i, 2] = -1.0
            poses_right[i, 0] = rx
            poses_right[i, 2] = -1.0
        else:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.scatter([lx, rx], [0, 0], c=['blue', 'red'])
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-0.1, 0.1)
            ax.axis('off')
            fig.savefig(os.path.join(output_dir, f"frame_{i:04d}.png"))
            plt.close(fig)

    if HAS_RENDER:
        for idx in range(len(frame_roll)):
            render_result_frame(poses_right, poses_left, frame_id=idx, idx_id="frame")
        # move produced images generated in `figs/` to output directory
        for f in os.listdir("figs"):
            shutil.move(os.path.join("figs", f), os.path.join(output_dir, f))

def main():
    parser = argparse.ArgumentParser(description="Generate simple hand position frames from MIDI")
    parser.add_argument('midi', help='Path to MIDI file')
    parser.add_argument('--out', default='demo_frames', help='Output directory')
    args = parser.parse_args()
    midi_to_frames(args.midi, args.out)

if __name__ == '__main__':
    main()
