# Rework proposal draft

I built a working proof for your visual search and automated packing brief using a real multi-variant fragrance catalog with visually similar bottles, labels, and fragile-item constraints.

The live demo lets your team upload an image, see ranked visual matches, verify an expected SKU, test deterministic barcode mismatch rejection, review a fragile-padded carton recommendation, and inspect monitoring evidence. The source includes validated catalog ingestion, SHA-256 deduplication, deterministic train/validation/test splits, PyTorch MobileNetV3 embeddings, a FastAPI service, confidence-and-margin safety gates, drift monitoring, tests, and a retraining/rollback playbook.

Current reproducible proof benchmark: 70 queries, 29.896 ms p95 hot search latency, 85.71% Recall@1, and 97.14% Recall@5 across 71 products and 485 variants. I treat your >99.9% SKU-verification and +10% conversion figures as production acceptance targets—not claims from a small demo. I would validate those targets against your warehouse images, barcode coverage, traffic baseline, and agreed test protocol.

Delivery mapping: Week 1 finalizes requirements and data audit; Weeks 2–4 build leak-resistant labeled splits and fine-tuned prototypes; Weeks 5–7 integrate and test search plus packing verification; Week 8 optimizes, documents, trains operators, and hands over the monitoring/retraining process. ERP/CRM integration and hardware procurement remain outside scope as requested.

Live demo: `[CLOUD_RUN_URL]`

Source and evidence: `[GITHUB_URL]`
