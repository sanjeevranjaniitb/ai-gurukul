# Hardware Requirements

## Minimum Specifications

| Resource | Minimum |
|----------|---------|
| RAM | 8 GB |
| CPU | 4-core (x86_64 or ARM64) |
| Disk | 10 GB free (models ~3.6 GB + Docker layers + runtime data) |
| OS | macOS / Linux |

## Peak Memory Budget

| Component | Limit |
|-----------|-------|
| Ollama (LLM) | 3 GB |
| Backend (FastAPI + models) | 2 GB |
| Frontend (nginx) | 256 MB |
| **Total peak** | **< 6 GB** |

## Model Disk Usage

| Model | Size | Quantization |
|-------|------|-------------|
| Llama 3.2 3B | ~2.0 GB | Q4_K_M (INT4) |
| Wav2Lip GAN checkpoint | ~416 MB | — |
| GFPGAN face detection weights | ~300 MB | — |
| Piper TTS (en_US-lessac-medium) | ~60 MB | ONNX |
| all-MiniLM-L6-v2 | ~80 MB | — |
| **Total** | **~2.9 GB** | |

## Notes

- All models use quantized or optimized variants to fit within edge constraints.
- No GPU required; all inference runs on CPU.
- Docker Compose enforces memory limits per service.
