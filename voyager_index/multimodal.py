"""
Multimodal model metadata and provider helpers for voyager-index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MultimodalModelSpec:
    plugin_name: str
    model_id: str
    architecture: str
    embedding_style: str
    modalities: tuple[str, ...]
    pooling_task: str
    serve_command: str


DEFAULT_MULTIMODAL_MODEL = "collfm2"


SUPPORTED_MULTIMODAL_MODELS: dict[str, MultimodalModelSpec] = {
    "collfm2": MultimodalModelSpec(
        plugin_name="collfm2",
        model_id="VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
        architecture="LFM2-VL + ColPali-style pooling",
        embedding_style="colpali",
        modalities=("text", "image"),
        pooling_task="token_embed",
        serve_command="vllm serve VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 --trust-remote-code --dtype bfloat16 --port 8200",
    ),
    "colqwen3": MultimodalModelSpec(
        plugin_name="colqwen3",
        model_id="VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
        architecture="Qwen3-VL + ColPali-style pooling",
        embedding_style="colpali",
        modalities=("text", "image"),
        pooling_task="token_embed",
        serve_command="vllm serve VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 --trust-remote-code --dtype bfloat16 --port 8200",
    ),
    "nemotron_colembed": MultimodalModelSpec(
        plugin_name="nemotron_colembed",
        model_id="nvidia/nemotron-colembed-vl-4b-v2",
        architecture="Qwen3-VL bidirectional + token-level ColBERT-style output",
        embedding_style="colbert",
        modalities=("text", "image"),
        pooling_task="token_embed",
        serve_command="vllm serve nvidia/nemotron-colembed-vl-4b-v2 --trust-remote-code --dtype bfloat16 --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200",
    ),
}

DEFAULT_MULTIMODAL_MODEL_SPEC = SUPPORTED_MULTIMODAL_MODELS[DEFAULT_MULTIMODAL_MODEL]


class VllmPoolingProvider:
    """
    Thin client for multimodal pooling endpoints exposed by vLLM-based providers.

    The provider intentionally stays generic: callers control the input payload
    and optional `extra_kwargs`, while the helper standardizes the request shape.
    """

    def __init__(self, endpoint: str, model: str, timeout: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def build_payload(
        self,
        input_data: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }
        if extra_kwargs:
            payload["extra_kwargs"] = dict(extra_kwargs)
        payload.update(kwargs)
        return payload

    def pool(
        self,
        input_data: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for VllmPoolingProvider. Install voyager-index[multimodal]."
            ) from exc

        payload = self.build_payload(input_data, extra_kwargs=extra_kwargs, **kwargs)
        response = httpx.post(f"{self.endpoint}/v1/pooling", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def batch_pool(
        self,
        inputs: Iterable[Any],
        extra_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [self.pool(item, extra_kwargs=extra_kwargs, **kwargs) for item in inputs]


__all__ = [
    "DEFAULT_MULTIMODAL_MODEL",
    "DEFAULT_MULTIMODAL_MODEL_SPEC",
    "MultimodalModelSpec",
    "SUPPORTED_MULTIMODAL_MODELS",
    "VllmPoolingProvider",
]
