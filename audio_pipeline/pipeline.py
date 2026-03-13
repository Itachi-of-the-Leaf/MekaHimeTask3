import os
import uuid
import logging
import shutil
import tempfile
from typing import List, Optional
from pydub import AudioSegment
import torch
import torchaudio
import torchaudio.transforms as T

from audio_pipeline.core.separator import SpeakerSeparator
from audio_pipeline.core.embedder import VoiceprintEmbedder
from audio_pipeline.core.vector_db import SpeakerDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(self, db_path: str = "./speaker_chroma_db"):
        """
        Initializes the end-to-end audio separation and voiceprint ID pipeline.
        Commercial-Safe Upgrade: Using Silero VAD and NeMo TitaNet.
        """
        logger.info("Initializing AudioPipeline...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Check if second GPU exists for high-load scaling (e.g., 5090 cluster)
        if torch.cuda.device_count() > 1:
            self.device = "cuda:1"
            
        self.separator = SpeakerSeparator()
        self.embedder = VoiceprintEmbedder()
        self.db = SpeakerDB(db_path=db_path)
        self.stranger_count = 0
        
        # Initialize Silero VAD (snakers4/silero-vad)
        logger.info("Loading Silero VAD model (Commercial-Safe)...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                               model='silero_vad', 
                                               force_reload=False)
        (self.get_speech_timestamps, _, _, _, _) = utils
        self.vad_model.to(self.device)
        
        logger.info(f"Pipeline initialized successfully on {self.device}.")
        
    def enroll_speaker_from_audio(self, name: str, audio_path: str, speaker_id: Optional[str] = None) -> str:
        """
        Refined Enrollment: Chunks audio into 5-second segments to store multiple vectors.
        Uses NeMo TitaNet for embeddings.
        """
        logger.info(f"Enrolling speaker '{name}' (multi-sample) with NeMo TitaNet: {audio_path}")
        
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 5000  # 5 seconds
        last_id = speaker_id
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
                end_ms = min(start_ms + chunk_length_ms, len(audio))
                if (end_ms - start_ms) < 2000:
                    continue
                    
                chunk = audio[start_ms:end_ms]
                chunk_path = os.path.join(temp_dir, f"{name}_chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                
                try:
                    embedding = self.embedder.embed_audio(chunk_path)
                    last_id = self.db.enroll_speaker(name=name, embedding=embedding, speaker_id=speaker_id)
                except Exception as e:
                    logger.error(f"Failed to enroll sample {i} for {name}: {e}")
                    
        # Hardware Hygiene: Clear VRAM after enrollment pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                    
        logger.info(f"Successfully finished TitaNet enrollment for '{name}'.")
        return last_id

    def _get_confidence_label(self, distance: float) -> str:
        """
        Confidence mapping for TitaNet-L embeddings.
        """
        if distance < 0.25:
            return "for sure"
        elif 0.25 <= distance <= 0.45:
            return "probably"
        else:
            return "likely"

    def _format_duration(self, seconds: float) -> str:
        """
        Formats duration into 'Xmin Ysec' or 'Xs'.
        """
        if seconds >= 60:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}min {secs}sec"
        else:
            return f"{seconds:.1f}s"
            
    def _gpu_resample(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Loads audio directly as a tensor and resamples using GPU.
        Returns flattened [time] tensor.
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        
        if sample_rate != target_sr:
            resampler = T.Resample(sample_rate, target_sr).to(self.device)
            waveform = resampler(waveform)
            
        # Ensure mono and flatten to [time] for Silero/NeMo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)
            
        return waveform

    def process_mixed_audio(self, mixed_audio_path: str, output_dir: str = "separated_outputs", threshold: float = 0.6) -> Optional[List[dict]]:
        """
        Enterprise Pipeline: Separate -> VAD Duration (Silero) -> Identification (TitaNet).
        """
        logger.info(f"Processing mixed audio (Enterprise Hybrid): {mixed_audio_path}")
        
        # 1. GPU-Accelerated Loading & Resampling
        try:
            waveform = self._gpu_resample(mixed_audio_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Failed to load/resample audio: {e}")
            return None
            
        # 2. Silero VAD Pre-Filter (Early Exit)
        speech_timestamps = self.get_speech_timestamps(waveform, 
                                                       self.vad_model, 
                                                       sampling_rate=16000,
                                                       threshold=0.5)
            
        if not speech_timestamps:
            logger.info("No human speech detected. Skipping heavy AI processing.")
            return None
            
        # Hardware Hygiene: Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Separation (The Engine)
        separated_files = self.separator.separate_audio(mixed_audio_path, output_dir=output_dir)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = []
        for file_path in separated_files:
            # 4. Silero VAD for Metadata Duration (Commercial-Safe)
            active_speech_duration = 0.0
            try:
                # Load separated track for VAD
                sep_waveform = self._gpu_resample(file_path, target_sr=16000)
                segments = self.get_speech_timestamps(sep_waveform, 
                                                      self.vad_model, 
                                                      sampling_rate=16000,
                                                      threshold=0.5)
                for seg in segments:
                    active_speech_duration += (seg['end'] - seg['start']) / 16000.0
            except Exception as e:
                logger.error(f"Failed to calculate Silero duration for {file_path}: {e}")
                # Fallback to total duration
                sig, fs = torchaudio.load(file_path)
                active_speech_duration = sig.shape[1] / fs
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 5. Identification (NeMo TitaNet)
            embedding = self.embedder.embed_audio(file_path)
            matches = self.db.identify_speaker(embedding, threshold=1.0)
            
            speaker_name = "Unknown"
            confidence_tag = "likely"
            best_dist = 1.0
            
            if matches:
                best_id, best_name, best_dist = matches[0]
                if best_dist <= threshold:
                    speaker_name = best_name
                    confidence_tag = self._get_confidence_label(best_dist)
                else:
                    self.stranger_count += 1
                    speaker_name = f"Stranger {self.stranger_count}"
                    confidence_tag = "likely"
            else:
                self.stranger_count += 1
                speaker_name = f"Stranger {self.stranger_count}"
                confidence_tag = "likely"
            
            # Hardware Hygiene: Final cache clear after ID
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            formatted_dur = self._format_duration(active_speech_duration)
            logger.info(f"{speaker_name} {confidence_tag} detected for {formatted_dur}")

            # 7. Auto-Enrollment Logic (New Intelligence)
            # If identified as Stranger/Unknown AND duration > 3.0s, enroll them
            if (speaker_name == "Unknown" or "Stranger" in speaker_name) and active_speech_duration > 3.0:
                auto_name = f"Auto_Stranger_{uuid.uuid4().hex[:6]}"
                logger.info(f"Auto-Enrolling significant new voice as '{auto_name}'...")
                try:
                    # Enrich DB with this new voiceprint using the clean isolated track
                    self.enroll_speaker_from_audio(auto_name, file_path)
                    speaker_name = auto_name
                    confidence_tag = "for sure" # Since we just enrolled them from this exact audio
                except Exception as e:
                    logger.error(f"Auto-enrollment failed for {auto_name}: {e}")
            
            sanitized_name = speaker_name.replace(" ", "_")
            new_name = os.path.join(os.path.dirname(file_path), f"{sanitized_name}_separated.wav")
            try:
                os.rename(file_path, new_name)
                final_file_path = new_name
            except Exception as e:
                logger.error(f"Failed to rename {file_path}: {e}")
                final_file_path = file_path

            results.append({
                "audio_file": final_file_path,
                "name": speaker_name,
                "confidence": confidence_tag,
                "distance": best_dist,
                "duration": active_speech_duration,
                "formatted_duration": formatted_dur,
                "matches": [{"name": m[1], "distance": m[2]} for m in matches]
            })
            
        return results
