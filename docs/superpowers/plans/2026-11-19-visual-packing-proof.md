# Visual Search and Automated Packing Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and publish a live Stone Wick Visual Search and Automated Packing proof with reproducible metrics.

**Architecture:** One FastAPI service serves an interactive UI and API. A build pipeline creates validated catalog artifacts; PyTorch provides image embeddings and search; deterministic packing checks, abstention, carton selection, and bounded monitoring complete the proof.

**Tech Stack:** Python 3.11, FastAPI, PyTorch/torchvision, Pillow, NumPy, scikit-learn, pytest, Docker, Google Cloud Run, GitHub

**Spec:** `docs/superpowers/specs/2026-11-19-visual-packing-proof-design.md`

## Global constraints

- Publish no credentials or private data.
- Report only generated metrics with sample counts and limitations.
- Label `<200 ms` as hot-path latency unless end-to-end timing is separately measured.
- Label `>99.9% accuracy` and `+10% conversion` as production targets until client validation.
- Wrong barcodes reject and uncertain visual matches hold for review.
- Cloud Run scales to zero unless minimum-instance cost is explicitly accepted.

## Tasks

1. Foundation and catalog: test and implement schema validation, shared-image grouping, deterministic split manifests, rejected-asset reporting, and a current Stone Wick source manifest.
2. PyTorch search and benchmark: test and implement normalized embeddings, unique ranking, empty-index handling, deterministic camera perturbations, and evidence generation.
3. Packing verification: test and implement barcode precedence, confidence and margin gates, human-review holds, orientation-aware carton fitting, and fragile padding.
4. Monitoring, API, and UI: test and implement bounded metrics, upload validation, all reviewer routes, and the responsive public demo.
5. Handover and verification: add documentation checks, container configuration, architecture/data/retraining docs, a measured proposal, full tests, local HTTP smoke test, and fresh review.
6. Publish and submit: publish GitHub source, deploy public Cloud Run with scale to zero, run live checks, update links, and submit the Rework application with evidence.

## Review focus

- Corrupt or non-image assets are rejected cleanly.
- Shared images never create duplicate ranked products.
- Duplicate hashes never cross dataset splits.
- Invalid uploads and empty indexes return bounded errors.
- Protective dimensions never claim a carton fit that is impossible.
