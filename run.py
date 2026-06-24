"""Entry point — run directly after activating the venv: python run.py"""
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

ENV = "dev"  # change to "prod" for production

import os
os.environ["APP_ENV"] = ENV
load_dotenv(Path(__file__).parent / "backend" / f".env.{ENV}")

if __name__ == "__main__":
    from backend.config import get_config

    config = get_config()
    print(f"Starting MisterPilot backend [{ENV}] on {config.server.host}:{config.server.port}")

    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=(ENV == "prod"),
    )
