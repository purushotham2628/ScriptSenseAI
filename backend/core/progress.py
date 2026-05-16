from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from fastapi import WebSocket


class ProgressHub:
    """In-memory WebSocket progress fanout for OCR and training jobs."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=100))

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[job_id].add(websocket)
        for event in self._history[job_id]:
            await websocket.send_json(event)

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        self._connections[job_id].discard(websocket)

    def publish_threadsafe(self, job_id: str, event: dict[str, Any]) -> None:
        self._history[job_id].append(event)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.publish(job_id, event))

    async def publish(self, job_id: str, event: dict[str, Any]) -> None:
        stale = []
        for websocket in list(self._connections[job_id]):
            try:
                await websocket.send_json({"job_id": job_id, **event})
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(job_id, websocket)


progress_hub = ProgressHub()
