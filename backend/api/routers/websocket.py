import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/progress/{job_id}")
async def training_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        for stage in ["queued", "validating", "preprocessing", "training", "evaluating", "checkpointing", "completed"]:
            await websocket.send_json({"job_id": job_id, "stage": stage})
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
