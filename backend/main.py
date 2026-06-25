from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import get_config
from .logging_config import setup_logging
from .api.routes import terminal, agent, model
from .api.routes.model import ChatCompletionRequest, model_chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield
    try:
        from .mcp.client import get_mcp_manager
        get_mcp_manager().stop_all()
    except Exception:
        pass


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title="MisterPilot API",
        description="AI coding assistant backend",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(terminal.router, prefix="/terminal", tags=["terminal"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])
    app.include_router(model.router, prefix="/model", tags=["model"])

    # OpenAI-compatible alias
    @app.post("/v1/chat/completions", tags=["v1"])
    async def v1_chat_completions(body: ChatCompletionRequest, raw_request: Request):
        """Drop-in replacement for OpenAI /v1/chat/completions."""
        return await model_chat(body, raw_request)

    return app


app = create_app()
