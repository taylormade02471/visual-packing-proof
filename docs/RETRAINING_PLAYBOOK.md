# Monitoring and retraining playbook

## Monitor

Track p50, p95, and p99 search latency; Recall@1 and Recall@5 on a stable labeled holdout; packing false-pass rate; barcode coverage; human-review rate; and embedding drift. Alert when p95 hot-path latency exceeds 200 ms, hold rate rises materially above its rolling baseline, centroid similarity crosses the configured drift floor, or any confirmed wrong-SKU shipment occurs. Conversion lift must be measured through an agreed A/B test, not inferred from retrieval metrics.

## Collect and label

Retain low-confidence and human-review frames with order/SKU identifiers after applying the client's privacy and retention policy. Review them in batches, correct labels against the SKU master, and include hard negatives such as identical vessels with different scent labels or sizes. Deduplicate by content hash and isolate related capture bursts to one split to prevent leakage.

## Train and evaluate

Rebuild the manifest, fine-tune the PyTorch encoder with metric learning on the training split, tune gates only on validation data, then freeze thresholds. A candidate cannot replace production unless it improves the agreed recall/precision measures, stays below the latency budget, and passes packing safety tests. The >99.9% requirement should be evaluated as end-to-end verified shipment accuracy with barcode and abstention behavior included, not as a claim from this small proof catalog.

## Release and rollback

Build an immutable container tagged with the commit and artifact version. Deploy a no-traffic Cloud Run revision, run smoke and load tests, then shadow or canary it before shifting traffic. Keep the prior revision and embedding artifact available. Roll back immediately if latency, false-pass, hold-rate, or error-rate guardrails regress. Record the dataset hash, model version, thresholds, benchmark report, approver, and rollback outcome in the release log.
