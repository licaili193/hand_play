import os
import numpy as np
import soundfile as sf
import librosa
import warnings
from typing import Union, Optional, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioProcessor:
    """
    Audio processing module for PianoMotion10M model.
    Handles audio file loading and preprocessing for inference.
    """
    
    def __init__(self, sample_rate: int = 16000, target_duration: float = 4.0, use_full_sequence: bool = False):
        """
        Initialize AudioProcessor.
        
        Args:
            sample_rate: Target sample rate for audio processing (default: 16000)
            target_duration: Target duration in seconds (default: 4.0) - only used if use_full_sequence=False
            use_full_sequence: If True, use full audio length; if False, truncate/pad to target_duration
        """
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.use_full_sequence = use_full_sequence
        self.target_samples = int(target_duration * sample_rate) if not use_full_sequence else None
    

    
    def load_wav_file(self, wav_path: str) -> np.ndarray:
        """
        Load and preprocess WAV file.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Try librosa first (handles various formats)
            audio_wave, sr = librosa.load(wav_path, sr=self.sample_rate)
            print(f"WAV loaded via librosa: {len(audio_wave)/sr:.2f}s at {sr}Hz")
            return audio_wave.astype(np.float32)
            
        except Exception as e:
            print(f"Librosa loading failed: {e}")
            try:
                # Fallback to soundfile
                audio_wave, sr = sf.read(wav_path)
                if len(audio_wave.shape) > 1:
                    # Convert stereo to mono
                    audio_wave = np.mean(audio_wave, axis=1)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    audio_wave = librosa.resample(audio_wave, orig_sr=sr, target_sr=self.sample_rate)
                
                print(f"WAV loaded via soundfile: {len(audio_wave)/self.sample_rate:.2f}s at {self.sample_rate}Hz")
                return audio_wave.astype(np.float32)
                
            except Exception as e2:
                print(f"Soundfile loading also failed: {e2}")
                # Return a reasonable fallback duration
                fallback_duration = 4.0 if not self.use_full_sequence else 10.0
                return np.zeros(int(fallback_duration * self.sample_rate), dtype=np.float32)
    
    def preprocess_audio(self, audio_wave: np.ndarray) -> np.ndarray:
        """
        Preprocess audio to match model requirements.
        
        Args:
            audio_wave: Input audio waveform
            
        Returns:
            Preprocessed audio waveform
        """
        current_samples = len(audio_wave)
        current_duration = current_samples / self.sample_rate
        
        if self.use_full_sequence:
            print(f"Audio preprocessing: {current_duration:.2f}s (full sequence mode)")
            # Use full sequence - only normalize, don't truncate/pad
            target_samples = current_samples
        else:
            print(f"Audio preprocessing: {current_duration:.2f}s -> {self.target_duration:.2f}s")
            # Handle duration mismatch - take first N seconds
            if current_samples > self.target_samples:
                # Take the first target_duration seconds
                audio_wave = audio_wave[:self.target_samples]
                print(f"  Taking first {self.target_duration:.2f}s of audio")
            elif current_samples < self.target_samples:
                # Pad with zeros if audio is shorter than target
                padding = self.target_samples - current_samples
                audio_wave = np.pad(audio_wave, (0, padding), mode='constant', constant_values=0)
                print(f"  Padding with {padding} zero samples")
            target_samples = self.target_samples
        
        # The Wav2Vec2Processor will handle normalization internally (z-score normalization)
        # We only need to ensure reasonable amplitude ranges to avoid numerical issues
        max_amplitude = np.max(np.abs(audio_wave))
        if max_amplitude > 10.0:  # Only normalize if amplitude is extremely large
            print(f"  Large amplitude detected ({max_amplitude:.2f}), normalizing to prevent numerical issues")
            audio_wave = audio_wave / max_amplitude
        elif max_amplitude < 1e-6:  # Handle near-silent audio
            print(f"  Very quiet audio detected ({max_amplitude:.2e}), applying small gain")
            audio_wave = audio_wave * 1000  # Small boost for very quiet audio
        
        return audio_wave.astype(np.float32)
    
    def process_audio_input(self, audio_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Process audio input from various sources (audio files only).
        
        Args:
            audio_input: Can be:
                - Path to audio file (.wav, .mp3, .flac, .m4a, .ogg)
                - Numpy array of audio samples
                
        Returns:
            Preprocessed audio waveform ready for model input
        """
        if isinstance(audio_input, str):
            # File path input
            file_ext = os.path.splitext(audio_input)[1].lower()
            
            if file_ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                print(f"Processing audio file: {audio_input}")
                audio_wave = self.load_wav_file(audio_input)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Only audio files are supported.")
                
        elif isinstance(audio_input, np.ndarray):
            # Direct numpy array input
            print("Processing numpy array input")
            audio_wave = audio_input.astype(np.float32)
            
            # Resample if necessary (only if not using full sequence)
            if not self.use_full_sequence and len(audio_wave) != self.target_samples:
                current_sr = len(audio_wave) / self.target_duration
                if current_sr != self.sample_rate:
                    audio_wave = librosa.resample(audio_wave, orig_sr=current_sr, target_sr=self.sample_rate)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Preprocess the audio
        processed_audio = self.preprocess_audio(audio_wave)
        
        print(f"[OK] Audio processing completed: {len(processed_audio)} samples ({len(processed_audio)/self.sample_rate:.2f}s)")
        return processed_audio
    
    def validate_audio(self, audio_wave: np.ndarray) -> bool:
        """
        Validate that audio meets model requirements.
        
        Args:
            audio_wave: Audio waveform to validate
            
        Returns:
            True if audio is valid, False otherwise
        """
        if not self.use_full_sequence and len(audio_wave) != self.target_samples:
            print(f"Warning: Audio length {len(audio_wave)} != expected {self.target_samples}")
            return False
        
        if not np.isfinite(audio_wave).all():
            print("Warning: Audio contains non-finite values")
            return False
        
        if np.max(np.abs(audio_wave)) > 1.0:
            print("Warning: Audio amplitude exceeds [-1, 1] range")
            return False
        
        return True

def create_audio_processor(sample_rate: int = 16000, target_duration: float = 4.0, use_full_sequence: bool = False) -> AudioProcessor:
    """
    Factory function to create an AudioProcessor instance.
    
    Args:
        sample_rate: Target sample rate
        target_duration: Target duration in seconds (only used if use_full_sequence=False)
        use_full_sequence: If True, use full audio length; if False, truncate/pad to target_duration
        
    Returns:
        AudioProcessor instance
    """
    return AudioProcessor(sample_rate=sample_rate, target_duration=target_duration, use_full_sequence=use_full_sequence) 