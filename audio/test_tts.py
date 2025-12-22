import torch
import torchaudio
import soundfile as sf
import sounddevice as sd
import numpy as np

# Monkey-patch torchaudio.load to use soundfile backend (required for Jetson/ARM64)
# This avoids the torchcodec dependency which isn't available on ARM64
_original_load = torchaudio.load

def _soundfile_load(filepath, *args, **kwargs):
    """Load audio using soundfile backend instead of torchcodec"""
    data, samplerate = sf.read(filepath, dtype='float32')
    # Convert to torch tensor and ensure correct shape (channels, samples)
    import torch
    if data.ndim == 1:
        data = data.reshape(1, -1)  # mono: (samples,) -> (1, samples)
    else:
        data = data.T  # stereo: (samples, channels) -> (channels, samples)
    return torch.from_numpy(data), samplerate

torchaudio.load = _soundfile_load

from TTS.api import TTS

def main():
    import os
    
    print("Initializing TTS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check if voice reference exists
    voice_ref = "voice_ref.wav"
    has_voice_ref = os.path.exists(voice_ref)
    
    if not has_voice_ref:
        print(f"\nWarning: {voice_ref} not found. Using default voice.")
        print("To use voice cloning, add a 6-15 second WAV file as 'voice_ref.wav'\n")
        
        # Use a simpler model that doesn't require voice cloning
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
        print("Model loaded successfully!")
        
        print("Generating speech...")
        tts.tts_to_file(
            text="Hey Sam — this is TTS running locally on your Jetson.",
            file_path="out.wav",
        )
    else:
        # Load XTTS v2 model for voice cloning
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("Model loaded successfully!")
        
        # Generate speech with voice cloning
        print("Generating speech with voice cloning...")
        tts.tts_to_file(
            text="Hey Sam — this is XTTS running locally with voice cloning.",
            speaker_wav=voice_ref,
            language="en",
            file_path="out.wav",
        )
    
    print("Speech generated successfully! Output saved to: out.wav")
    
    # Play the audio
    print("Playing audio...")
    audio_data, sample_rate = sf.read("out.wav")
    sd.play(audio_data, sample_rate)
    sd.wait()  # Wait until audio is finished playing
    print("Playback complete!")

if __name__ == "__main__":
    main()

