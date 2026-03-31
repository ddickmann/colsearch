"""Standalone FastAPI app exposing only the stateless fulfilment optimizer."""

from __future__ import annotations

from voyager_index._internal.inference.stateless_optimizer import create_stateless_optimizer_app

app = create_stateless_optimizer_app()
