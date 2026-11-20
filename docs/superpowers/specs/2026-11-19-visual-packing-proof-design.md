# Visual Search and Automated Packing Proof Design

Build a public, reviewer-testable proof for the Rework Visual Search and Automated Warehouse Packing brief using the user's Stone Wick/TaylorMade catalog. The proof demonstrates Python, PyTorch, FastAPI, cloud serving, labeled-data handling, monitoring, and handover capability without fabricating production results.

## Goals

- Serve one public Cloud Run site with visual search, SKU verification, carton selection, monitoring, and evidence views.
- Build a validated catalog manifest from public Stone Wick data already embedded in the user's site.
- Use a pretrained PyTorch image encoder and normalized in-memory cosine search.
- Produce reproducible hot-path latency, Recall@1, Recall@5, confidence, and hold-rate evidence.
- Reject wrong barcodes deterministically and send uncertain visual matches to `HUMAN_REVIEW_HOLD`.
- Publish clean source and handover documentation in GitHub.

## Boundaries

- No ERP/CRM, scanner, conveyor, camera-hardware, or ongoing maintenance integration.
- The proof does not claim a 10% conversion lift without production traffic and an experiment.
- The proof does not claim greater than 99.9% SKU verification accuracy without a representative client test set.
- Shared product images are grouped; repeated variants are not treated as independent visual evidence.
- Credentials, private exports, customer data, and hidden account details never enter source or evidence.

## Chosen approach

A single FastAPI service serves the UI and JSON API. Build-time ingestion validates catalog images and emits deterministic manifests. A pretrained MobileNetV3 PyTorch encoder creates L2-normalized embeddings and searches them with torch matrix multiplication. Packing verification combines optional barcode evidence, visual score and margin gates, abstention, fragile padding, and carton selection. Monitoring records bounded latency, confidence, hold, and drift signals.

Generic public fashion data was rejected because it would not reuse the user's real work. A hosted vector database was rejected because this proof catalog does not justify the extra cost and credentials.

## Acceptance behavior

- Sample selection or image upload returns ranked, unique products and measured model/search latency.
- A wrong barcode rejects before visual inference.
- An auto-pass requires the expected identity, score threshold, and top-two margin threshold.
- Ambiguous, unsupported, or low-confidence input returns `HUMAN_REVIEW_HOLD`.
- Carton output includes orientation-aware fit, fragile padding, and fill efficiency.
- Evidence includes sample count, environment, timestamp, metrics, and limitations.
- Tests cover catalog leakage, shared images, search, mismatch and hold decisions, carton fitting, monitoring, API validation, and the public page.

## Handover

The repository includes deployment instructions, architecture notes, an API contract, a data dictionary, the benchmark command, drift/retraining guidance, and an eight-week roadmap mapped to the client's four deliverables.
