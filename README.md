# Language VAE Prompt Injection Guardrail

A production-grade, defense-in-depth filter cascade using a Sentence/Language Variational Autoencoder for **unsupervised anomaly detection** against prompt injection attacks.

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: Structural Filter      │  < 5ms
│ RegEx · Length · Token Density   │
└──────────────┬──────────────────┘
               │ PASS
               ▼
┌─────────────────────────────────┐
│ Stage 2: Language VAE Core      │  15-35ms
│ Reconstruction Loss + Latent    │
│ Mahalanobis Distance            │
└──────────────┬──────────────────┘
               │ PASS
               ▼
┌─────────────────────────────────┐
│ Stage 3: Vector Guardrail       │  < 10ms
│ Cosine Similarity vs Attack DB  │
└──────────────┬──────────────────┘
               │ PASS
               ▼
         ✅ LLM Pipeline
```

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Train the VAE

```bash
python -m vae_guardrail.training.train --epochs 10
```

### 3. Calibrate Thresholds

```bash
python -m vae_guardrail.scoring.calibration --checkpoint checkpoints/vae_guardrail_best.pt
```

### 4. Start the API Server

```bash
python -m vae_guardrail.api.server
```

### 5. Validate a Prompt

```bash
curl -X POST http://localhost:8080/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I sort a list in Python?"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/validate` | POST | Validate a prompt through the cascade |
| `/v1/health` | GET | Health check |
| `/v1/metrics` | GET | Prometheus-compatible metrics |
| `/v1/config` | POST | Runtime threshold updates |
| `/dashboard` | GET | Monitoring dashboard |

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
SHADOW_MODE=true          # Log only (no blocking) for first 2 weeks
VAE_ANOMALY_THRESHOLD=1.24
VECTOR_SIMILARITY_THRESHOLD=0.85
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

Apache-2.0
