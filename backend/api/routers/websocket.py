from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.progress import progress_hub

router = APIRouter()


@router.websocket("/progress/{job_id}")
async def training_progress(websocket: WebSocket, job_id: str):
    await progress_hub.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_hub.disconnect(job_id, websocket)
