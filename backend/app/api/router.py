from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import analyses, auth, health
from app.core.config import get_settings

api_router = APIRouter()
# Health probes stay at the root so orchestrators can reach them without the API prefix.
api_router.include_router(health.router)
api_router.include_router(auth.router, prefix=get_settings().api_prefix)
api_router.include_router(analyses.router, prefix=get_settings().api_prefix)

__all__ = ["api_router"]