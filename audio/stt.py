import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from typing import Optional
import tempfile
import os


class SpeechToText:
    """
    Speech-to-Text engine using faster-whisper.
    
    Optimized for Jetson with CUDA acceleration.
    """
    
    def __init__(self, model_size: str = "base.en", use_cuda: bool = False, compute_type: str = "int8"):
        """
        Initialize Speech-to-Text engine.
        
        Args:
            model_size: Whisper model size. Options: "tiny.en", "base.en", "small.en", "medium.en"
                       Recommended: "base.en" for good balance of speed/accuracy
            use_cuda: Whether to use CUDA if available
            compute_type: Computation precision. Options: "float16", "int8", "float32"
        """
        self.model_size = model_size
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        
        device = "cpu"  # Default to CPU
        actual_compute_type = "int8"  # CPU-friendly compute type
        
        # Try CUDA if requested
        if use_cuda:
            try:
                print(f"Attempting to load Whisper model '{model_size}' on CUDA...")
                self.model = WhisperModel(
                    model_size,
                    device="cuda",
                    compute_type=compute_type
                )
                print("Whisper model loaded successfully on CUDA!")
                return
            except (ValueError, Exception) as e:
                print(f"CUDA not available ({e}), falling back to CPU...")
                device = "cpu"
        
        print(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=actual_compute_type
        )
        print("Whisper model loaded successfully!")
    
    def listen(self, duration: float = 5.0, silence_threshold: float = 0.01, 
               silence_duration: float = 1.5) -> np.ndarray:
        """
        Record audio from microphone with simple silence detection.
        
        Args:
            duration: Maximum recording duration in seconds
            silence_threshold: Audio amplitude threshold to detect silence (0.0 to 1.0)
            silence_duration: How long silence must persist to stop recording
            
        Returns:
            Audio data as numpy array
        """
        print("Recording... (speak now)")
        
        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Simple silence detection to trim end
        audio = audio.flatten()
        
        # Find last point where audio exceeds threshold
        above_threshold = np.abs(audio) > silence_threshold
        if np.any(above_threshold):
            # Find the last significant audio
            last_sound_idx = np.where(above_threshold)[0][-1]
            # Add a small buffer after last sound
            buffer_samples = int(0.5 * self.sample_rate)
            end_idx = min(len(audio), last_sound_idx + buffer_samples)
            audio = audio[:end_idx]
        
        print("Recording complete!")
        return audio
    
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            
        Returns:
            Transcribed text
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,  # Use voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Combine all segments
        text = " ".join([segment.text.strip() for segment in segments])
        return text.strip()
    
    def transcribe_array(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio numpy array to text.
        
        Args:
            audio: Audio data as numpy array
            language: Language code (default: "en")
            
        Returns:
            Transcribed text
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, self.sample_rate)
        
        try:
            text = self.transcribe(tmp_path, language=language)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return text
    
    def listen_and_transcribe(self, duration: float = 5.0, language: str = "en") -> str:
        """
        Record audio and transcribe in one step.
        
        Args:
            duration: Maximum recording duration in seconds
            language: Language code (default: "en")
            
        Returns:
            Transcribed text
        """
        audio = self.listen(duration=duration)
        return self.transcribe_array(audio, language=language)


# Example usage
if __name__ == "__main__":
    # Initialize STT
    stt = SpeechToText(model_size="base.en")
    
    print("\n=== Speech-to-Text Test ===")
    print("You will be recorded for 5 seconds. Speak clearly!\n")
    
    # Test listen and transcribe
    text = stt.listen_and_transcribe(duration=5.0)
    
    print(f"\nTranscription: {text}")


