import argparse
import os
import numpy as np
from mido import MidiFile
import matplotlib.pyplot as plt

# Simplified utilities adapted from PianoMotion10M datasets/utils.py

def read_midi(midi_path):
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat
    assert len(midi_file.tracks) == 2
    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second
    message_list = []
    ticks = 0
    time_in_second = []
    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)
    return np.array(message_list), np.array(time_in_second)

def read_frame_roll(midi_events_time, midi_events, start_time, segment_seconds, fps):
    for bgn_idx, event_time in enumerate(midi_events_time):
        if event_time > start_time:
            break
    for fin_idx, event_time in enumerate(midi_events_time):
        if event_time > start_time + segment_seconds:
            break
    note_events = []
    buffer_dict = {}
    for i in range(bgn_idx, fin_idx):
        attr = midi_events[i].split(' ')
        if attr[0] in ['note_on', 'note_off']:
            midi_note = int(attr[2].split('=')[1])
            velocity = int(attr[3].split('=')[1])
            if attr[0] == 'note_on' and velocity > 0:
                buffer_dict[midi_note] = {
                    'onset_time': midi_events_time[i],
                    'velocity': velocity
                }
            else:
                if midi_note in buffer_dict:
                    note_events.append({
                        'midi_note': midi_note,
                        'onset_time': buffer_dict[midi_note]['onset_time'],
                        'offset_time': midi_events_time[i],
                        'velocity': buffer_dict[midi_note]['velocity']
                    })
                    del buffer_dict[midi_note]
    for midi_note in buffer_dict:
        note_events.append({
            'midi_note': midi_note,
            'onset_time': buffer_dict[midi_note]['onset_time'],
            'offset_time': start_time + segment_seconds,
            'velocity': buffer_dict[midi_note]['velocity']
        })
    frames_num = int(round(segment_seconds * fps))
    frame_roll = np.zeros((frames_num, 88), dtype=np.float32)
    for note_event in note_events:
        piano_note = np.clip(note_event['midi_note'] - 21, 0, 87)
        if 0 <= piano_note <= 87:
            bgn_frame = int(round((note_event['onset_time'] - start_time) * fps))
            fin_frame = int(round((note_event['offset_time'] - start_time) * fps))
            if fin_frame >= 0:
                frame_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = 1
    return frame_roll

def note_to_x(note_index):
    return -1.0 + 2.0 * (note_index / 87.0)

def midi_to_frames(midi_path, output_dir, fps=30):
    events, times = read_midi(midi_path)
    duration = times[-1] if len(times) > 0 else 0
    frame_roll = read_frame_roll(times, events, 0, duration, fps)
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frame_roll):
        notes = np.where(frame > 0)[0]
        left = [n for n in notes if n < 44]
        right = [n for n in notes if n >= 44]
        lx = np.mean([note_to_x(n) for n in left]) if left else -0.5
        rx = np.mean([note_to_x(n) for n in right]) if right else 0.5
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.scatter([lx, rx], [0, 0], c=['blue', 'red'])
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.1, 0.1)
        ax.axis('off')
        fig.savefig(os.path.join(output_dir, f"frame_{i:04d}.png"))
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate simple hand position frames from MIDI")
    parser.add_argument('midi', help='Path to MIDI file')
    parser.add_argument('--out', default='demo_frames', help='Output directory')
    args = parser.parse_args()
    midi_to_frames(args.midi, args.out)

if __name__ == '__main__':
    main()
