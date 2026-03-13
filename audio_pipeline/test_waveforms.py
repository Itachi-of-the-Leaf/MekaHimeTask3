import torchaudio
import torch
import os

print("\n--- Verifying Separated Output Audio Waveforms ---")

separated_dir = "test_audio/separated"
if not os.path.exists(separated_dir):
    print("Error: Separated audio directory not found!")
    exit(1)

files = [f for f in os.listdir(separated_dir) if f.endswith(".wav")]
if not files:
    print("Error: No separated wav files found!")
    exit(1)

for f in files:
    path = os.path.join(separated_dir, f)
    try:
        waveform, sr = torchaudio.load(path)
        print(f"\nAnalyzing File: {path}")
        max_amp = torch.max(torch.abs(waveform)).item()
        variance = torch.var(waveform).item()
        print(f"  -> Max Amplitude: {max_amp:.4f}")
        print(f"  -> Variance (energy var): {variance:.6f}")
        
        # Check if variance and amplitude represent valid audio and not just flatline/garbage
        if max_amp > 1e-3 and variance > 1e-5:
            print("  -> Status: VALID AUDIO WAVEFORM DETECTED (Not static/silent)")
        else:
            print("  -> Status: INVALID (Static/Silent)")
    except Exception as e:
        print(f"  -> Error loading {path}: {e}")

print("\nVerification Complete.")
