"""Uvicorn entrypoint."""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings


def main() -> None:
    settings.ensure_dirs()
    uvicorn.run(
        "backend.app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.host == "127.0.0.1",
    )


if __name__ == "__main__":
    main()
