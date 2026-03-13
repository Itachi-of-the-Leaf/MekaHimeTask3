import torch
import torchaudio
import logging
from typing import List
import nemo.collections.asr as nemo_asr
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

class VoiceprintEmbedder:
    def __init__(self, model_name: str = "titanet_large"):
        """
        Initializes the NVIDIA NeMo TitaNet embedder.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            self.device = "cuda:1"
            
        logger.info(f"Loading NeMo TitaNet ({model_name}) on {self.device}...")
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def embed_audio(self, audio_path: str) -> List[float]:
        """
        Extracts a voiceprint embedding from the given audio file using NeMo TitaNet.
        Manually handles loading to ensure correct (batch, time) shape.
        """
        logger.info(f"Extracting TitaNet embedding for {audio_path}")
        
        try:
            # Load and resample to 16kHz
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(self.device)
            
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000).to(self.device)
                waveform = resampler(waveform)
            
            # Ensure mono and shape is (batch, time) -> (1, time)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            # Ensure no trailing singleton dims
            if waveform.ndim == 3:
                waveform = waveform.squeeze(-1)
                
            length = torch.tensor([waveform.shape[1]]).to(self.device)
            
            with torch.no_grad():
                # EncDecSpeakerLabelModel.forward returns (logits, embeddings)
                _, embedding = self.model.forward(input_signal=waveform, input_signal_length=length)
                
            # Hardware Hygiene
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return embedding.flatten().cpu().tolist()
            
        except Exception as e:
            logger.error(f"TitaNet embedding failed for {audio_path}: {e}")
            raise
