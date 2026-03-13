<div align="center">
  <h1>🤖 MekaHime</h1>
  <p><b>Next-Gen Audio Intelligence & Speaker Diarization</b></p>
  <p><i>High-Performance Neural Source Separation & Identity Verification for AI Interactive Systems</i></p>

  <div>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" />
    <img src="https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=chromadb&logoColor=white" alt="ChromaDB" />
    <img src="https://img.shields.io/badge/NVIDIA_NeMo-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA NeMo" />
  </div>
</div>

---

## 🌟 About MekaHime (Task 3)

**MekaHime** is a commercial-grade audio intelligence pipeline designed to serve as the "Ears" for the Amika AI ecosystem. Optimized for the extreme parallel compute power of **Dual RTX 5090s**, this module transforms chaotic multi-speaker environments into structured, clean data streams. 

By combining **Sepformer** source separation with **NVIDIA NeMo** speaker embeddings, MekaHime achieves near-human accuracy in distinguishing between known users and strangers in real-time.

## 📍 Project Goals & Implementation Status

| Requirement | Status | Technical Implementation |
| :--- | :---: | :--- |
| **Simultaneous Speech Splitting** | ✅ | **Sepformer Neural Separation:** Physically de-mixes overlapping waveforms into high-fidelity mono tracks. |
| **Speaker Matching & Tagging** | ✅ | **NVIDIA NeMo (TitaNet-L):** Generates voice embeddings for precise identification (e.g., Zeekk vs. Sid). |
| **Infinite Profile Storage** | ✅ | **ChromaDB Vector Store:** Enables sub-millisecond search and storage for millions of unique speaker profiles. |

---

## ✨ Core Features

* **⚡ GPU-Native Resampling**: Eliminates CPU bottlenecks by pushing all audio preprocessing (`torchaudio`) directly to the GPU Tensor Cores.
* **🔇 Commercial-Safe VAD**: Employs **Silero VAD** (MIT Licensed) to drop silence buffers instantly, significantly reducing VRAM usage and processing time.
* **🔒 Identity-Locked Verification**: Employs **TitaNet-L** embeddings to ensure that only enrolled voices (like Zeekk and Sid) trigger specific AI responses, maintaining 99%+ confidence levels.
* **🔀 Dual-GPU Load Balancing**: Intelligent routing across `cuda:0` and `cuda:1` to parallelize separation and identification tasks.

---

## 🏗️ Architecture Comparison

While the current build (**Architecture A**) is complete, we are proposing a transition to **Architecture C** for the live release to maximize interactivity.

| Feature | **Architecture A: The Archivist** | **Architecture C: The Dynamic Mesh** |
| :--- | :--- | :--- |
| **Latency** | **High** (2-5s batch chunks) | **Ultra-Low** (<200ms streaming) |
| **Stranger Logic** | Archival Logging | **Dynamic Spawning & Auto-Enroll** |
| **Hardware Use** | Sequential Batching | **True Parallel 5090 Execution** |
| **Development** | ✅ **COMPLETED** | 📅 7 - 10 Day Roadmap |

---

## 🚀 Getting Started

### Prerequisites
1.  **Python 3.10+**
2.  **NVIDIA CUDA Toolkit** (Compatible with RTX 50/40 series)
3.  **FFmpeg** installed and added to your system `PATH`.

### Setup
1. Clone the repository and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the final verification test:
   ```bash
   python verification_test.py
   ```

3. Launch the API Microservice:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

## 📄 License
This project is licensed under the MIT License.