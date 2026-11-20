from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from visual_packing.catalog import parse_bundle_catalog


STORE_URL = "https://stone-wick.com/"
USER_AGENT = "StoneWick-Visual-Packing-Proof/0.1"


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> None:
    homepage = fetch_text(STORE_URL)
    match = re.search(r'<script[^>]+src="(?P<src>/assets/index-[^"]+\.js)"', homepage)
    if not match:
        raise RuntimeError("Stone Wick application bundle was not found")
    products = parse_bundle_catalog(fetch_text(STORE_URL.rstrip("/") + match.group("src")))
    destination = Path("data/catalog.source.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([product.model_dump(mode="json") for product in products], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(products)} products to {destination}")


if __name__ == "__main__":
    main()
