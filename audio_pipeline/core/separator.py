import os
import logging
import torch
import torchaudio
import numpy as np
from typing import List
from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.separation import SepformerSeparation

logger = logging.getLogger(__name__)

class SpeakerSeparator:
    def __init__(self, model_source: str = "speechbrain/sepformer-whamr", savedir: str = "pretrained_models/sepformer-whamr"):
        """
        Initializes the source separator using SpeechBrain's Sepformer.
        """
        self.model_source = model_source
        self.savedir = savedir
        logger.info(f"Loading Sepformer from {model_source}...")
        
        # Monkeypatch for torchaudio compatibility if needed
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['sox', 'soundfile']
            
        self.model = SepformerSeparation.from_hparams(
            source=model_source, 
            savedir=savedir,
            run_opts={"device":"cuda"} if torch.cuda.is_available() else None
        )
        
    def separate_audio(self, mixed_audio_path: str, output_dir: str = "separated_outputs") -> List[str]:
        """
        Separates a mixed audio file into isolated tracks.
        Returns a list of paths to the separated audio files.
        """
        logger.info(f"Separating sources from {mixed_audio_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Separate
        est_sources = self.model.separate_file(path=mixed_audio_path)
        
        # Save each source
        base_name = os.path.splitext(os.path.basename(mixed_audio_path))[0]
        separated_files = []
        
        # est_sources is [time, batch, src] or similar depending on version, 
        # but separate_file usually returns [time, src] for a single file.
        # We need to transpose to [src, time] for saving if it's [time, src]
        if est_sources.dim() == 2:
            est_sources = est_sources.unsqueeze(0) # [1, time, src]
            
        # [batch, time, src] -> [src, time]
        num_sources = est_sources.shape[2]
        for i in range(num_sources):
            source = est_sources[:, :, i] # [1, time]
            output_path = os.path.join(output_dir, f"{base_name}_source_{i}.wav")
            torchaudio.save(output_path, source.cpu(), 8000) # Sepformer WHAMR is usually 8k
            logger.info(f"Saved separated source {i} to {output_path}")
            separated_files.append(output_path)
            
        return separated_files

    def diarize(self, mixed_audio_path: str, output_dir: str = "separated_outputs") -> List[dict]:
        """
        Legacy diarization method. In 'True Hybrid', we use pyannote on separated tracks.
        """
        return []
