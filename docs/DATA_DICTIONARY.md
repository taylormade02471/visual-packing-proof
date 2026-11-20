# Data dictionary

## Source catalog

`data/catalog.source.json` is a normalized snapshot of publicly displayed Stone Wick products. Each product has a stable product identifier, name, category, source image URL, and variants. Each variant includes a SKU, display title, dimensions in centimeters, and a fragile flag. Dimensions are proof defaults inferred from the format name and must be replaced with client-authoritative master data before production carton decisions.

## Visual assets

`artifacts/catalog.json` groups products whose downloaded image bytes share the same SHA-256 digest. `asset_id` is that digest; it prevents byte-identical images from leaking across dataset splits or being indexed repeatedly. A visual asset may therefore reference multiple product IDs, names, categories, and variants. `sample_url` exists only for the curated reviewer gallery.

## Dataset manifest

`artifacts/dataset_manifest.json` records deterministic train, validation, and test membership at the deduplicated asset level. It includes accepted and rejected counts and the reason for every rejected file. The current catalog is a proof dataset; client images must be split by physical capture session or item batch where possible to avoid near-duplicate leakage.

## Embeddings and benchmark

`artifacts/embeddings.pt` stores normalized MobileNetV3 feature vectors aligned by row with `artifacts/catalog.json`. `artifacts/benchmark_report.json` records encoder identity, environment, query count, latency percentiles, recall metrics, scope, and limitations. Latency covers the warm model plus in-memory search path; network transit and cold starts are excluded and must be measured separately during production load testing.
