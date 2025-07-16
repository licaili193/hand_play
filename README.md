# Hand Play Demo

This repository contains a simple demonstration of generating hand pose frames from a MIDI file. The demo reuses MIDI processing code from the [PianoMotion10M](https://github.com/agnJason/PianoMotion10M) project.

## Quick Start

1. Install dependencies (mido, matplotlib, numpy):
   ```bash
   pip install mido matplotlib numpy
   ```
2. Place a MIDI file in the repository, e.g. `example.mid`.
3. Run the demo script:
   ```bash
   python demo/midi_to_frames.py example.mid --out output_frames
   ```

Images for each frame will be saved in the specified output directory. Each image shows an approximate location of the left (blue) and right (red) hands over time.

This is a minimal example that uses a small portion of the PianoMotion10M codebase. It does not require the full dataset or pretrained models.
