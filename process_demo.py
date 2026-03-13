import os
import sys

# Adjust path if script is run from inside or outside MekaHimeTask3
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio_pipeline'))

import torch
import torchaudio

# Monkeypatch for SpeechBrain <-> Torchaudio > 2.1 compatibility
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['sox', 'soundfile']

from pipeline import AudioPipeline

def main():
    print("Initializing the AudioPipeline...")
    # Initialize the Pipeline
    pipeline = AudioPipeline()
    
    mixed_audio_path = "audio_pipeline/demo_input/overlapping_voices.mp3"
    if not os.path.exists(mixed_audio_path):
        mixed_audio_path = "test_audio/overlapping_voices.mp3"

    print(f"Processing mixed audio: {mixed_audio_path}")
    
    # Process Audio and save to demo_results
    results = pipeline.process_mixed_audio(
        mixed_audio_path=mixed_audio_path,
        output_dir="demo_results"
    )
    
    print("\n--- Pipeline Results ---")
    for res in results:
        print(f"File: {res['audio_file']}")
        if res.get('matches'):
            best_match = res['matches'][0]
            print(f" -> Speaker identified: {best_match['name']} (Distance: {best_match['distance']:.4f})")
        else:
            print(" -> Speaker identified: Unknown")

if __name__ == "__main__":
    main()
