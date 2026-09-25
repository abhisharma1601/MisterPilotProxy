from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, debug
from .config import get_config
from .logging_config import setup_logging
from .api.routes.model import ChatCompletionRequest, model_chat
from .services import cost_service
from .services.model_access import available_models, is_auto_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield
    # Let in-flight wallet charges finish; any still pending after the timeout
    # is cancelled and written to the dead-letter file, not lost.
    await cost_service.drain()


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

    # OpenAI-compatible endpoints.
    @app.get("/v1/models", tags=["v1"])
    async def v1_models(raw_request: Request):
        """Drop-in replacement for OpenAI /v1/models: the models the caller's key can use."""
        auth = raw_request.headers.get("Authorization", "")
        raw_key = auth[7:].strip() if auth.startswith("Bearer ") else ""
        return {"object": "list", "data": available_models(raw_key)}

    @app.post("/v1/chat/completions", tags=["v1"])
    async def v1_chat_completions(body: ChatCompletionRequest, raw_request: Request):
        """Drop-in replacement for OpenAI /v1/chat/completions."""
        # Scorer calibration printout (DEBUG_ROUTE_REQUEST=1). misterpilot-auto
        # reports the verdict it routes with from model_access instead, so
        # nothing is scored twice.
        if debug.enabled() and not is_auto_model(body.model):
            debug.score_in_background(body.model_dump(exclude_none=True), body.model)
        return await model_chat(body, raw_request)

    return app


app = create_app()
