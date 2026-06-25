"""Entry point — run directly after activating the venv: python run.py"""
import os
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

load_dotenv(Path(__file__).parent / "backend" / ".env")
ENV = os.environ.get("APP_ENV", "dev")

if __name__ == "__main__":
    from backend.config import get_config

    config = get_config()
    print(f"Starting MisterPilot backend [{ENV}] on {config.server.host}:{config.server.port}")

    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=(ENV == "dev"),
    )
