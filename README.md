# Stone Wick Visual Search & Packing AI Proof

A live-review proof for the Rework brief **Visual Search and Automated Packing Systems for Digital Commerce and Fulfillment**. It uses the public Stone Wick Fragrance Studio catalog to demonstrate image retrieval, SKU safety gates, fragile-item carton selection, monitoring, and a reviewer-friendly web interface.

## Live proof

- **Live Cloud Run demo:** https://visual-packing-proof-915634103558.us-central1.run.app
- **45-second screen recording:** [visual-packing-proof-demo.mp4](visual-packing-proof-demo.mp4)
- **Cloud project:** `project-5b166851-62cd-439e-a82`

## Measured proof evidence

The checked-in benchmark contains 71 products, 485 variants, and 70 unique visual assets. On 70 camera-style perturbed queries using a warm MobileNetV3 model and an in-memory normalized embedding index, the proof measured:

- p95 search latency: **42.073 ms** (target: under 200 ms)
- Recall@1: **85.71%**
- Recall@5: **97.14%**
- Rejected or corrupt catalog assets: **0**

These are reproducible proof results, not production promises. The brief's **>99.9% SKU verification** and 10% conversion-lift figures remain production targets that require the client's labeled warehouse camera data, barcode coverage, traffic baseline, and acceptance thresholds. The verifier protects fulfillment precision by preferring deterministic barcode checks and routing uncertain visual cases to `HUMAN_REVIEW_HOLD` instead of guessing.

## Run locally

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
$env:PYTHONPATH="src"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m uvicorn visual_packing.api:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. Choose a catalog sample or upload an image, select an expected SKU, and run visual search plus packing verification.

## API

- `GET /health` — service readiness and indexed counts
- `GET /api/catalog` — safe demo samples
- `GET /api/evidence` — checked-in benchmark evidence
- `GET /api/metrics` — drift, confidence, decision, and latency counters
- `POST /api/search` — multipart image visual retrieval
- `POST /api/packing/verify` — multipart image, expected SKU, and optional barcode SKU

## Rebuild evidence

`scripts/refresh_catalog.py` refreshes the owner-controlled public catalog source. `scripts/build_artifacts.py` validates and deduplicates images, creates deterministic train/validation/test splits, builds multi-view embeddings, and writes the benchmark report. Rebuilding downloads public catalog images and the official torchvision MobileNetV3 weights.

## Deployment

The `Dockerfile` is Cloud Run compatible and honors the `PORT` environment variable. Deploy it as a new service such as `visual-packing-proof`; do not replace an unrelated service. Scale-to-zero is appropriate for a portfolio proof. Cold-start time is separate from the measured hot inference/search path.

See [architecture](docs/ARCHITECTURE.md), [data dictionary](docs/DATA_DICTIONARY.md), [retraining playbook](docs/RETRAINING_PLAYBOOK.md), and [Rework proposal](docs/REWORK_PROPOSAL.md).
