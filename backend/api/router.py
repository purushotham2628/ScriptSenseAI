from fastapi import APIRouter

from backend.api.routers import active_learning, auth, datasets, inference, training, websocket

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(active_learning.router, prefix="/active-learning", tags=["active-learning"])
api_router.include_router(websocket.router, prefix="/ws", tags=["websocket"])
