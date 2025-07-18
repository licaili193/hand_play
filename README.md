# Hand Play Demo

This repository contains a simple demonstration of generating hand pose frames and videos from a MIDI file. The demo reuses MIDI processing code from the [PianoMotion10M](https://github.com/agnJason/PianoMotion10M) project.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install mido matplotlib numpy imageio moviepy soundfile
   ```

2. Place a MIDI file in the repository, e.g. `example.mid`.

3. Run the demo script:
   ```bash
   python midi_to_frames.py example.mid --out output_frames
   ```

## Features

- **Frame Generation**: Creates individual PNG images for each frame showing hand poses
- **Video Generation**: Automatically creates an MP4 video from the frames (default: `hand_animation.mp4`)
- **Smart Hand Assignment**: Intelligently determines if a piece is single-hand or two-hand and assigns notes accordingly
- **Realistic Finger Movement**: Generates natural finger joint angles based on note patterns

## Command Line Options

```bash
python midi_to_frames.py <midi_file> [options]

Options:
  --out OUTPUT_DIR     Output directory (default: output)
  --fps FPS           Frames per second (default: 30)
  --no-video          Skip video generation
  --video-name NAME   Output video filename (default: hand_animation.mp4)
```

## Output

- **Frames**: Individual PNG images named `smart_frame_XXXX.png`
- **Video**: MP4 video file showing the hand animation
- **Analysis**: Console output showing piece type detection and processing status

## Examples

```bash
# Basic usage with default settings
python midi_to_frames.py examples/example_1.mid

# Custom output directory and FPS
python midi_to_frames.py examples/example_1.mid --out my_output --fps 60

# Generate frames only (no video)
python midi_to_frames.py examples/example_1.mid --no-video

# Custom video filename
python midi_to_frames.py examples/example_1.mid --video-name my_animation.mp4
```

This is a minimal example that uses a small portion of the PianoMotion10M codebase. It does not require the full dataset or pretrained models.
