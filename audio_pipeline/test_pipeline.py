import os
import torch
import torchaudio

# Monkeypatch for SpeechBrain <-> Torchaudio > 2.1 compatibility
# MUST be applied before any speechbrain import
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['sox', 'soundfile']

import logging
from pipeline import AudioPipeline

logging.basicConfig(level=logging.INFO)

def create_dummy_audio(filename: str, freq: float = 440.0, duration: float = 3.0, sample_rate: int = 16000):
    """Generates a simple sine wave dummy audio file."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Simple sine wave
    waveform = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)
    torchaudio.save(filename, waveform, sample_rate)
    return filename

def main():
    os.makedirs("test_audio", exist_ok=True)
    
    # 1. Generate dummy enrollments for two "speakers"
    freq_zeekk = 200.0  # low frequency for zeekk
    freq_sid = 800.0    # higher frequency for sid
    
    zeekk_audio = create_dummy_audio("test_audio/zeekk_enrollment.wav", freq=freq_zeekk)
    sid_audio = create_dummy_audio("test_audio/sid_enrollment.wav", freq=freq_sid)
    
    # Generate a "mixed" track by just adding them
    t = torch.linspace(0, 3.0, int(16000 * 3.0))
    waveform_mixed = (torch.sin(2 * torch.pi * freq_zeekk * t) + 
                      torch.sin(2 * torch.pi * freq_sid * t)).unsqueeze(0)
    waveform_mixed = waveform_mixed / torch.max(torch.abs(waveform_mixed))
    mixed_audio_wav = "test_audio/mixed_audio.wav"
    torchaudio.save(mixed_audio_wav, waveform_mixed, 16000)
    
    # Convert to MP3 for testing
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(mixed_audio_wav)
    mixed_audio_mp3 = "test_audio/mixed_audio.mp3"
    audio.export(mixed_audio_mp3, format="mp3")
    mixed_audio = mixed_audio_mp3
    
    # 2. Initialize pipeline
    # The models will take some time to download on the first run.
    pipeline = AudioPipeline(db_path="./test_chroma_db")
    
    # 3. Enroll speakers
    pipeline.enroll_speaker_from_audio(name="Zeekk", audio_path=zeekk_audio)
    pipeline.enroll_speaker_from_audio(name="Sid", audio_path=sid_audio)
    
    # 4. Process mixed audio
    # The threshold might need tuning for dummy sine waves (not real voices)
    # but the pipeline mechanics will be verified.
    results = pipeline.process_mixed_audio(mixed_audio, output_dir="test_audio/separated", threshold=1.0)
    
    print("\n--- Final Results ---")
    for res in results:
        print(f"File: {res['audio_file']}")
        if res['matches']:
            best = res['matches'][0]
            print(f" -> Speaker: {best['name']} (Distance: {best['distance']:.4f})")
        else:
            print(" -> Speaker: Unknown")

if __name__ == "__main__":
    main()
