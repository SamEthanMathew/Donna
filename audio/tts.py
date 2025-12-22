import torch
import torchaudio
import soundfile as sf
import sounddevice as sd
import numpy as np
from typing import Optional

# Monkey-patch torchaudio.load to use soundfile backend (required for Jetson/ARM64)
def _soundfile_load(filepath, *args, **kwargs):
    """Load audio using soundfile backend instead of torchcodec"""
    data, samplerate = sf.read(filepath, dtype='float32')
    if data.ndim == 1:
        data = data.reshape(1, -1)  # mono: (samples,) -> (1, samples)
    else:
        data = data.T  # stereo: (samples, channels) -> (channels, samples)
    return torch.from_numpy(data), samplerate

torchaudio.load = _soundfile_load

from TTS.api import TTS


class TextToSpeech:
    """
    Text-to-Speech engine for the Jetson Agent.
    
    Supports two modes:
    1. Default voice (Tacotron2) - fast, no voice cloning
    2. Voice cloning (XTTS v2) - clones a reference voice
    """
    
    def __init__(self, voice_ref_path: Optional[str] = None, use_cuda: bool = False):
        """
        Initialize TTS engine.
        
        Args:
            voice_ref_path: Path to reference voice WAV file (6-15s) for voice cloning.
                          If None, uses default voice.
            use_cuda: Whether to use CUDA if available.
        """
        self.voice_ref_path = voice_ref_path
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        
        # Load appropriate model
        if voice_ref_path:
            print(f"Loading XTTS v2 for voice cloning on {self.device}...")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            self.use_voice_cloning = True
        else:
            print(f"Loading Tacotron2 on {self.device}...")
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(self.device)
            self.use_voice_cloning = False
        
        print("TTS model loaded successfully!")
    
    def speak(self, text: str, save_path: Optional[str] = None, block: bool = True):
        """
        Convert text to speech and play it.
        
        Args:
            text: The text to speak.
            save_path: Optional path to save the audio file. If None, saves to temp file.
            block: Whether to wait for audio playback to finish.
        """
        # Use temp file if no save path specified
        if save_path is None:
            save_path = "/tmp/tts_output.wav"
        
        # Generate speech
        if self.use_voice_cloning:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_ref_path,
                language="en",
                file_path=save_path,
            )
        else:
            self.tts.tts_to_file(
                text=text,
                file_path=save_path,
            )
        
        # Play the audio
        audio_data, sample_rate = sf.read(save_path)
        sd.play(audio_data, sample_rate)
        
        if block:
            sd.wait()  # Wait until playback finishes
    
    def generate(self, text: str, save_path: str):
        """
        Generate speech and save to file without playing.
        
        Args:
            text: The text to convert to speech.
            save_path: Path to save the audio file.
        """
        if self.use_voice_cloning:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_ref_path,
                language="en",
                file_path=save_path,
            )
        else:
            self.tts.tts_to_file(
                text=text,
                file_path=save_path,
            )


# Example usage
if __name__ == "__main__":
    # Initialize TTS with default voice
    tts = TextToSpeech()
    
    # Speak some text
    tts.speak("Hello! I am your Jetson agent, ready to assist you.")


