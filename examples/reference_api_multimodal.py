"""
Minimal multimodal reference API example with precomputed patch embeddings.

This follows the default OSS production path: a disk-backed local collection
using the exact FP16 Triton MaxSim path unless an optional quantized profile is
enabled explicitly.
"""

from __future__ import annotations

import httpx


BASE_URL = "http://127.0.0.1:8080"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        client.post(
            "/collections/demo-mm",
            json={
                "dimension": 4,
                "kind": "multimodal",
            },
        )
        client.post(
            "/collections/demo-mm/points",
            json={
                "points": [
                    {
                        "id": "page-1",
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "payload": {"doc_id": "invoice.pdf", "page_number": 1},
                    },
                    {
                        "id": "page-2",
                        "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
                        "payload": {"doc_id": "report.pdf", "page_number": 2},
                    },
                ]
            },
        )
        response = client.post(
            "/collections/demo-mm/search",
            json={
                "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                "top_k": 2,
            },
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()
