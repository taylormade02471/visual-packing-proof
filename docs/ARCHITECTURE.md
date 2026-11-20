# Architecture

The proof separates offline catalog preparation from the online reviewer path. Offline, the catalog adapter parses Stone Wick's public product bundle, validates records with Pydantic, downloads image bytes, rejects unreadable files, deduplicates by SHA-256, and creates deterministic train, validation, and test manifests. A torchvision MobileNetV3-Large encoder produces three camera-style views per visual asset. Their normalized embeddings are averaged and stored with catalog metadata.

Online, FastAPI loads the encoder and checked-in embedding tensor once at process startup. `/api/search` validates the upload, computes one normalized query embedding, and performs an in-memory cosine-similarity search with PyTorch matrix multiplication. This is the path measured by the latency benchmark. The web dashboard consumes the same public API, so reviewers test the implementation rather than a mocked animation.

Packing verification is deliberately risk-aware. An available barcode is authoritative: an exact match passes and a mismatch rejects. Without a barcode, a visual match passes only when the expected asset is top-ranked and both score and margin gates are satisfied. Ambiguous cases become `HUMAN_REVIEW_HOLD`. The carton selector evaluates item orientations and adds fragile-item clearance before selecting the smallest fitting standard carton.

The in-process monitor records latency, confidence, drift similarity, and packing decisions. In production these records would be exported to Cloud Monitoring or a warehouse event stream. Client ERP/CRM integration and hardware procurement are outside this proof, matching the brief exclusions.
