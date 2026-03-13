# MekaHime Task 3: Enterprise Audio Pipeline

State-of-the-art Neural Source Separation and Speaker Identification microservice.

## Core Features
- **True Hybrid Architecture**: Separates overlapping voices first for mathematically clean speaker identification.
- **Enterprise Embeddings**: Powered by NVIDIA NeMo TitaNet-L for massive scalability and commercial-safe voiceprints.
- **Efficient VAD**: Silero VAD (Commercial-Safe) for early silence exit and precise duration calculation.
- **Auto-Enrollment**: Automatically identifies and learns new recurring voices (Strangers).
- **GPU-Native**: Fully optimized for CUDA with aggressive hardware hygiene.
- **API Microservice**: Exposed via FastAPI for high-performance integration.

## Installation
1. Ensure CUDA is installed and available.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API
Start the microservice:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Usage
Submit mixed audio for processing:
```bash
curl -X 'POST' \
  'http://localhost:8000/process_audio' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_audio_mix.wav'
```

## Verification
Run the comprehensive test suite:
```bash
python verification_test.py
```
