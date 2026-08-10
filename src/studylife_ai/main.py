"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI

from studylife_ai.api import chat, health
from studylife_ai.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(title=settings.app_name)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
