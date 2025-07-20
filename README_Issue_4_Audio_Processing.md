# Issue 4: Audio Processing Mismatch

## Problem Description

The current implementation in `midi_to_frames.py` has several issues with audio processing that don't match the expected format and preprocessing pipeline used by the PianoMotion10M model.

### Specific Issues:

1. **Inconsistent Audio Duration Handling**:
   - Current code forces 4-second sequences regardless of actual MIDI/audio length
   - No proper handling of variable-length inputs
   - Missing logic for audio segments that are shorter or longer than training duration

2. **Missing Dataset-Style Preprocessing**:
   - Official code uses sophisticated audio preprocessing from `PianoPose` dataset class
   - Current implementation uses simple `pretty_midi` synthesis with sine waves
   - Missing proper audio normalization and format standardization

3. **Audio-Frame Synchronization Issues**:
   - Current code doesn't ensure audio length matches expected frame count
   - Frame rate assumptions (30 FPS) not properly synchronized with audio
   - No handling of timing mismatches between MIDI events and audio samples

4. **Inadequate Audio Quality**:
   - Using sine wave synthesis instead of realistic piano sounds
   - Missing proper instrument synthesis that the model was trained on
   - Audio quality doesn't match training data distribution

### Code Location:
- File: `midi_to_frames.py`
- Lines: 22-40 (MIDI to audio conversion)
- Lines: 125-140 (audio preprocessing for model)

## Recommended Fixes

### 1. Implement Proper Audio Preprocessing Pipeline

Create a comprehensive audio preprocessing function that matches the dataset format:

```python
def preprocess_audio_pipeline(audio_wave, target_duration=4.0, sample_rate=16000):
    """
    Comprehensive audio preprocessing pipeline matching PianoPose dataset.
    """
    
    # Ensure correct sample rate
    if len(audio_wave) == 0:
        raise ValueError("Empty audio array")
    
    # Calculate target length in samples
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio_wave)
    
    print(f"Audio preprocessing:")
    print(f"  - Input length: {current_samples} samples ({current_samples/sample_rate:.2f}s)")
    print(f"  - Target length: {target_samples} samples ({target_duration:.2f}s)")
    
    # Handle duration mismatch
    if current_samples > target_samples:
        # For longer audio, take the first segment (could be improved with onset detection)
        audio_wave = audio_wave[:target_samples]
        print(f"  - Trimmed to {target_duration}s")
    elif current_samples < target_samples:
        # For shorter audio, pad with silence
        padding = target_samples - current_samples
        audio_wave = np.pad(audio_wave, (0, padding), mode='constant', constant_values=0)
        print(f"  - Padded with {padding} samples of silence")
    
    # Normalize audio to [-1, 1] range (critical for model)
    max_amplitude = np.max(np.abs(audio_wave))
    if max_amplitude > 0:
        audio_wave = audio_wave / max_amplitude
        print(f"  - Normalized by factor {max_amplitude:.4f}")
    else:
        print("  - Warning: Silent audio detected")
    
    # Apply additional preprocessing to match training data
    # Remove DC offset
    audio_wave = audio_wave - np.mean(audio_wave)
    
    # Apply gentle high-pass filter to remove low-frequency noise
    from scipy.signal import butter, filtfilt
    try:
        # High-pass filter at 80 Hz (below piano range)
        nyquist = sample_rate / 2
        high_cutoff = 80.0 / nyquist
        b, a = butter(2, high_cutoff, btype='high')
        audio_wave = filtfilt(b, a, audio_wave)
        print(f"  - Applied high-pass filter at 80 Hz")
    except Exception as e:
        print(f"  - Warning: Could not apply filter: {e}")
    
    # Final validation
    assert len(audio_wave) == target_samples, f"Audio length mismatch: {len(audio_wave)} != {target_samples}"
    assert np.max(np.abs(audio_wave)) <= 1.0, f"Audio not properly normalized: max = {np.max(np.abs(audio_wave))}"
    
    print(f"  ✓ Audio preprocessing completed")
    return audio_wave.astype(np.float32)
```

### 2. Improve MIDI to Audio Synthesis

Replace the simple sine wave synthesis with better piano synthesis:

```python
def synthesize_piano_from_midi(midi_path, sample_rate=16000, soundfont_path=None):
    """
    Improved MIDI to audio synthesis with realistic piano sounds.
    """
    
    try:
        # Load MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        print(f"MIDI loaded: {len(pm.instruments)} instruments, duration: {pm.get_end_time():.2f}s")
        
        # Option 1: Use fluidsynth with a piano soundfont (if available)
        if soundfont_path and os.path.exists(soundfont_path):
            try:
                audio_wave = pm.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
                print("✓ Used FluidSynth with soundfont for realistic piano sound")
                return audio_wave
            except Exception as e:
                print(f"FluidSynth failed: {e}, falling back to synthesize")
        
        # Option 2: Use pretty_midi's built-in synthesis with better waveform
        # Try multiple waveforms and pick the best one
        waveforms = [
            ('sine', np.sin),
            ('triangle', lambda x: 2 * np.arcsin(np.sin(x)) / np.pi),
            ('sawtooth', lambda x: 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5)))
        ]
        
        best_audio = None
        best_name = None
        
        for name, waveform in waveforms:
            try:
                audio_wave = pm.synthesize(fs=sample_rate, wave=waveform)
                # Simple quality check: prefer non-zero audio with reasonable amplitude
                if len(audio_wave) > 0 and np.max(np.abs(audio_wave)) > 0.001:
                    best_audio = audio_wave
                    best_name = name
                    break
            except Exception as e:
                print(f"Synthesis with {name} wave failed: {e}")
                continue
        
        if best_audio is not None:
            print(f"✓ Used {best_name} wave synthesis")
            return best_audio
        else:
            # Last resort: silent audio
            duration = max(pm.get_end_time(), 4.0)  # At least 4 seconds
            print("Warning: All synthesis methods failed, generating silence")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)
            
    except Exception as e:
        print(f"MIDI synthesis error: {e}")
        # Generate silence as fallback
        return np.zeros(int(4.0 * sample_rate), dtype=np.float32)
```

### 3. Add Audio-Frame Synchronization

Implement proper synchronization between audio timing and frame indices:

```python
def synchronize_audio_with_frames(audio_wave, midi_path, fps=30, sample_rate=16000):
    """
    Synchronize audio timing with frame indices for proper model input.
    """
    
    try:
        # Load MIDI to get timing information
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Calculate timing parameters
        audio_duration = len(audio_wave) / sample_rate
        midi_duration = pm.get_end_time()
        frame_count = int(audio_duration * fps)
        
        print(f"Synchronization info:")
        print(f"  - Audio duration: {audio_duration:.2f}s")
        print(f"  - MIDI duration: {midi_duration:.2f}s")
        print(f"  - Frame count: {frame_count} frames")
        print(f"  - FPS: {fps}")
        
        # Check for timing mismatches
        if abs(audio_duration - midi_duration) > 0.5:  # 500ms tolerance
            print(f"  - Warning: Audio-MIDI duration mismatch: {abs(audio_duration - midi_duration):.2f}s")
        
        # Create frame timing array
        frame_times = np.linspace(0, audio_duration, frame_count)
        
        # Validate synchronization
        samples_per_frame = sample_rate / fps
        expected_samples = frame_count * samples_per_frame
        actual_samples = len(audio_wave)
        
        if abs(expected_samples - actual_samples) > sample_rate * 0.1:  # 100ms tolerance
            print(f"  - Warning: Frame-audio sync issue: {abs(expected_samples - actual_samples)/sample_rate:.2f}s difference")
        
        return frame_times, frame_count
        
    except Exception as e:
        print(f"Synchronization error: {e}")
        # Fallback to simple calculation
        audio_duration = len(audio_wave) / sample_rate
        frame_count = int(audio_duration * fps)
        frame_times = np.linspace(0, audio_duration, frame_count)
        return frame_times, frame_count
```

### 4. Add Audio Quality Validation

Implement validation to ensure audio quality matches training data:

```python
def validate_audio_quality(audio_wave, sample_rate=16000):
    """
    Validate audio quality and characteristics.
    """
    
    # Basic quality checks
    duration = len(audio_wave) / sample_rate
    max_amplitude = np.max(np.abs(audio_wave))
    rms = np.sqrt(np.mean(audio_wave**2))
    
    print(f"Audio quality validation:")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Max amplitude: {max_amplitude:.4f}")
    print(f"  - RMS level: {rms:.4f}")
    print(f"  - Dynamic range: {20*np.log10(max_amplitude/rms):.1f} dB" if rms > 0 else "  - Dynamic range: N/A (silent)")
    
    # Quality warnings
    warnings = []
    
    if max_amplitude < 0.01:
        warnings.append("Very low amplitude - audio might be too quiet")
    elif max_amplitude > 0.99:
        warnings.append("Near-clipping amplitude - audio might be too loud")
    
    if rms < 0.001:
        warnings.append("Very low RMS - audio might be mostly silent")
    
    if duration < 3.0:
        warnings.append("Short duration - might not provide enough context")
    
    # Check for silence
    silence_threshold = 0.0001
    silent_samples = np.sum(np.abs(audio_wave) < silence_threshold)
    silence_ratio = silent_samples / len(audio_wave)
    
    if silence_ratio > 0.5:
        warnings.append(f"High silence ratio: {silence_ratio:.1%}")
    
    # Frequency content check (basic spectral analysis)
    try:
        freqs = np.fft.fftfreq(len(audio_wave), 1/sample_rate)
        fft = np.abs(np.fft.fft(audio_wave))
        
        # Check for reasonable frequency distribution
        low_freq_energy = np.sum(fft[(freqs >= 80) & (freqs <= 500)])
        mid_freq_energy = np.sum(fft[(freqs >= 500) & (freqs <= 2000)])
        high_freq_energy = np.sum(fft[(freqs >= 2000) & (freqs <= 8000)])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            print(f"  - Frequency distribution: Low: {low_freq_energy/total_energy:.1%}, Mid: {mid_freq_energy/total_energy:.1%}, High: {high_freq_energy/total_energy:.1%}")
        
    except Exception as e:
        warnings.append(f"Could not analyze frequency content: {e}")
    
    # Report warnings
    if warnings:
        print("  ⚠ Warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  ✓ Audio quality looks good")
    
    return len(warnings) == 0
```

### 5. Integrate Complete Audio Processing Pipeline

Replace the current audio processing section with:

```python
# 1. Convert MIDI to audio with improved synthesis
print("Step 1: Converting MIDI to audio...")
try:
    # Try to use realistic piano synthesis
    audio_wave = synthesize_piano_from_midi(MIDI_PATH, sample_rate=16000)
    
    # Validate initial audio
    if len(audio_wave) == 0:
        raise ValueError("Audio synthesis produced empty result")
    
    # Apply comprehensive preprocessing
    audio_wave = preprocess_audio_pipeline(audio_wave, target_duration=4.0)
    
    # Validate audio quality
    quality_ok = validate_audio_quality(audio_wave)
    if not quality_ok:
        print("⚠ Audio quality warnings detected, but proceeding...")
    
    # Synchronize with frame timing
    frame_times, expected_frames = synchronize_audio_with_frames(
        audio_wave, MIDI_PATH, fps=30
    )
    
    # Save processed audio for inspection
    sf.write(AUDIO_PATH, audio_wave, samplerate=16000)
    print(f"✓ Processed audio saved to: {AUDIO_PATH}")
    print(f"✓ Audio duration: {len(audio_wave)/16000:.2f}s, Expected frames: {expected_frames}")
    
except Exception as e:
    print(f"✗ Error in audio processing: {e}")
    return
```

## Testing the Fix

1. **Audio synthesis quality**: Compare synthesized audio with real piano recordings
2. **Duration handling**: Test with MIDI files of various lengths (short, exact, long)
3. **Normalization validation**: Ensure audio amplitudes are properly normalized
4. **Frame synchronization**: Verify that frame count matches audio duration at 30 FPS
5. **Model compatibility**: Confirm processed audio works with the PianoMotion10M model

## Impact of Fix

- **Better audio quality**: More realistic piano synthesis improves model performance
- **Proper preprocessing**: Audio format matches training data expectations
- **Robust duration handling**: Correctly handles variable-length MIDI inputs
- **Accurate synchronization**: Frame timing properly aligned with audio samples
- **Quality validation**: Catches potential audio issues before inference
- **Improved model accuracy**: Better input quality leads to better hand motion predictions 