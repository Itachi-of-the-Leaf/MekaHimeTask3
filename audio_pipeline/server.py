import os
import shutil
import tempfile
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pipeline import AudioPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MekaHime Audio Pipeline API")

# Initialize Pipeline (Singleton-ish)
# Models load once on startup
logger.info("Starting AudioPipeline for API service...")
pipeline = AudioPipeline()

class IdentificationResult(BaseModel):
    audio_file: str
    name: str
    confidence: str
    distance: float
    duration: float
    formatted_duration: str

@app.post("/process_audio", response_model=List[dict])
async def process_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, processes it through the True Hybrid pipeline,
    and returns identification logs and file paths.
    """
    logger.info(f"Received API request: {file.filename}")
    
    # Check file extension
    if not file.filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # Save upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Process the audio
        # Note: threshold is set to 0.6 as per enterprise spec
        results = pipeline.process_mixed_audio(tmp_path, threshold=0.6)
        
        if results is None:
            return [{"info": "No human speech detected. Processing skipped."}]
            
        return results

    except Exception as e:
        logger.error(f"API Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup the uploaded temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
