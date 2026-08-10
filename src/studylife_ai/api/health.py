"""Liveness/readiness endpoint."""

from fastapi import APIRouter

from studylife_ai.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()
