"""
Minimal late-interaction reference API example with precomputed embeddings.

This follows the default OSS production path: a disk-backed local collection
served through the exact-by-default Triton MaxSim stack.
"""

from __future__ import annotations

import httpx


BASE_URL = "http://127.0.0.1:8080"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        client.post(
            "/collections/demo-li",
            json={
                "dimension": 4,
                "kind": "late_interaction",
            },
        )
        client.post(
            "/collections/demo-li/points",
            json={
                "points": [
                    {
                        "id": "doc-1",
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "payload": {"text": "invoice total due"},
                    },
                    {
                        "id": "doc-2",
                        "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
                        "payload": {"text": "meeting transcript"},
                    },
                ]
            },
        )
        response = client.post(
            "/collections/demo-li/search",
            json={
                "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                "top_k": 2,
            },
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()
