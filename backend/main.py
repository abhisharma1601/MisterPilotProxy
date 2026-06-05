from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
from .logging_config import setup_logging
from .api.routes import chat, workspace, edit, terminal, agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title="MisterPilot API",
        description="AI coding assistant backend",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(workspace.router, prefix="/workspace", tags=["workspace"])
    app.include_router(edit.router, prefix="/edit", tags=["edit"])
    app.include_router(terminal.router, prefix="/terminal", tags=["terminal"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])

    return app


app = create_app()
