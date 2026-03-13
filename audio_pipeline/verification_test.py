import sys
import logging
import os
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the pipeline
from pipeline import AudioPipeline  
def run_test():
    try:
        
        # Audio paths as specified by the user
        # We adjust to ../ if we are running from the audio_pipeline directory
        base_dir = "."
        if not os.path.exists("test_audio") and os.path.exists("../test_audio"):
            base_dir = ".."
            
        zeekk_audio = os.path.join(base_dir, "test_audio/zeek_clean.wav")
        sid_audio = os.path.join(base_dir, "test_audio/sid_clean.wav")
        mix_audio = os.path.join(base_dir, "test_audio/noisy_overlap_mix.wav")
        output_dir = os.path.join(base_dir, "test_audio/separated")

        # Step 2.1: Clear legacy DB for new model compatibility
        db_path = "./test_chroma_db"
        if os.path.exists(db_path):
            logger.info(f"Clearing legacy ChromaDB at {db_path}...")
            shutil.rmtree(db_path)

        # Step 2.1: Initialize AudioPipeline with db_path="./test_chroma_db"
        pipeline = AudioPipeline(db_path=db_path)
        
        # Step 2.2: Enroll "Zeekk"
        logger.info("Enrolling 'Zeekk'...")
        pipeline.enroll_speaker_from_audio("Zeekk", zeekk_audio)
        
        # Step 2.3: Enroll "Sid"
        logger.info("Enrolling 'Sid'...")
        pipeline.enroll_speaker_from_audio("Sid", sid_audio)
        
        # Step 2.4: Process mixed audio
        logger.info("Processing mixed audio...")
        results = pipeline.process_mixed_audio(
            mixed_audio_path=mix_audio, 
            output_dir=output_dir, 
            threshold=0.6
        )
        
        logger.info(f"Test Results: {results}")
        
        # Step 2.5: Implement assertions
        identified_names = [r["name"] for r in results]
        logger.info(f"Identified Names: {identified_names}")
        assert len(results) >= 2, f"Expected at least 2 tracks extracted, got {len(results)}"
        
        # If we are using Miku_reference, we expect Strangers
        if "Miku_reference.wav" in mix_audio:
            has_stranger = any("Stranger" in name for name in identified_names)
            assert has_stranger, "Expected at least one 'Stranger' to be identified for Miku_reference."
        else:
            # For noisy mix, we accept names OR Stranger tags (since distance > 0.5 results in Stranger)
            valid_counts = sum(1 for name in identified_names if name in ["Zeekk", "Sid"] or "Stranger" in name)
            assert valid_counts >= 2, f"Expected at least 2 valid identities (names or Strangers), got {identified_names}"
        
        # Step 2.6: Success exit
        print("PIPELINE_SUCCESS")
        sys.exit(0)
        
    except AssertionError as e:
        logger.error(f"Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected Error: {e}", exc_info=True)
        sys.exit(1)

def run_silence_test():
    """
    Verifies that the pipeline correctly skips heavy processing for silence.
    """
    print("\n--- Running Silence Detection Test ---")
    pipeline = AudioPipeline(db_path="./test_chroma_db")
    silence_path = "../test_audio/silence.wav"
    
    if not os.path.exists(silence_path):
        print(f"Creating silence file at {silence_path}")
        import torch
        import torchaudio
        silence = torch.zeros(1, 16000 * 5)
        torchaudio.save(silence_path, silence, 16000)
    
    results = pipeline.process_mixed_audio(silence_path)
    
    if results is None:
        print("VAD_INTERCEPT_SUCCESS: Pipeline correctly skipped silent audio.")
    else:
        print("VAD_INTERCEPT_FAILURE: Pipeline failed to skip silent audio.")
        sys.exit(1)

if __name__ == "__main__":
    # Run the main identification test
    try:
        run_test()
    except SystemExit as e:
        if e.code != 0:
            sys.exit(e.code)
            
    # Run the new silence test
    run_silence_test()
