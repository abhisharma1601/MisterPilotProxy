from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import get_config
from .logging_config import setup_logging
from .api.routes import chat, workspace, edit, terminal, agent, model

_LANDING = (Path(__file__).parent / "static" / "index.html").read_text()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


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

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def landing():
        return _LANDING

    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(workspace.router, prefix="/workspace", tags=["workspace"])
    app.include_router(edit.router, prefix="/edit", tags=["edit"])
    app.include_router(terminal.router, prefix="/terminal", tags=["terminal"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])
    app.include_router(model.router, prefix="/model", tags=["model"])

    return app


app = create_app()
