"""
Voyager Index Reference API

FastAPI server for deploying the local voyager-index reference service.

Usage:
    # Development
    uvicorn voyager_index.server:app --reload --port 8080

    # Local service
    uvicorn voyager_index.server:app --host 127.0.0.1 --port 8080

Author: Latence Team
License: Apache-2.0
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .api.service import SearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Factory
# ============================================================================

def create_app(
    title: str = "Voyager Index Reference API",
    version: str = "0.1.0",
    enable_cors: bool = False,
    index_path: Optional[str] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        title: API title
        version: API version
        enable_cors: Enable CORS middleware
        index_path: Path for index storage

    Returns:
        Configured FastAPI application
    """

    index_dir = (
        index_path
        or os.environ.get("VOYAGER_INDEX_PATH")
        or os.environ.get("LATENCE_INDEX_PATH")
        or "/data/voyager-index"
    )
    os.makedirs(index_dir, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan events."""
        logger.info(f"Starting {title} v{version}")
        logger.info(f"Index directory: {index_dir}")
        app.state.search_service = SearchService(index_dir)

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Running on CPU")
        except ImportError:
            logger.info("PyTorch not available, running on CPU")

        yield

        # Shutdown
        close_service = getattr(app.state.search_service, "close", None)
        if callable(close_service):
            close_service()
        logger.info("Shutting down server")

    app = FastAPI(
        title=title,
        description="""
## Voyager Index Reference API

Local-first reference service for the open-source `voyager-index` runtime.

### Features

- Durable collection metadata on disk
- Dense, late-interaction, and multimodal collection types
- Hybrid dense+sparse retrieval for dense collections
- MaxSim-backed late-interaction retrieval
- Stateless fulfilment optimizer at `/reference/optimize` (canonical contract; needs `latence_solver`)
- Source-doc preprocessing at `/reference/preprocess/documents` for PDF, DOCX, XLSX, and image inputs
- Local multimodal retrieval over precomputed embeddings

### Quick Start

1. Render source docs into page images:
```bash
curl -X POST http://localhost:8080/reference/preprocess/documents \\
  -H "Content-Type: application/json" \\
  -d '{"source_paths": ["/data/source/invoice.pdf"]}'
```

2. Create a collection:
```bash
curl -X POST http://localhost:8080/collections/docs \\
  -H "Content-Type: application/json" \\
  -d '{"dimension": 128, "kind": "dense"}'
```

3. Add points:
```bash
curl -X POST http://localhost:8080/collections/docs/points \\
  -H "Content-Type: application/json" \\
  -d '{"points": [{"id": "1", "vector": [...]}]}'
```

4. Search:
```bash
curl -X POST http://localhost:8080/collections/docs/search \\
  -H "Content-Type: application/json" \\
  -d '{"vector": [...], "top_k": 10}'
```
        """,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.index_dir = index_dir

    # CORS middleware
    if enable_cors:
        allowed_origins = os.environ.get(
            "VOYAGER_CORS_ORIGINS",
            "http://127.0.0.1,http://localhost",
        ).split(",")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Authorization"],
        )

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # Include routes
    app.include_router(router, prefix="")

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8080))
    workers = int(os.environ.get("WORKERS", 1))
    if workers != 1:
        raise SystemExit(
            "WORKERS>1 is not supported by the voyager-index reference API. "
            "Run a single worker until shared-state coordination is implemented."
        )

    uvicorn.run(
        "voyager_index.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


# Default application instance
app = create_app()


if __name__ == "__main__":
    main()

