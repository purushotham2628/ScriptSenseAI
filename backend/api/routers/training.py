from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.core.config import get_settings
from backend.core.security import require_user_id
from backend.ml.training.trainer import create_training_job
from backend.schemas.platform import TrainingJobResponse, TrainingRequest

router = APIRouter()
TRAINING_JOBS = {}


def _run_training(job_id: str, manifest_path: Path) -> None:
    TRAINING_JOBS[job_id]["status"] = "running"
    try:
        result = create_training_job(manifest_path)
        TRAINING_JOBS[job_id].update({"status": "completed", "result": result})
    except Exception as exc:  # pragma: no cover - job status path
        TRAINING_JOBS[job_id].update({"status": "failed", "error": str(exc)})


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks, user_id: str = Depends(require_user_id)):
    settings = get_settings()
    manifest_path = settings.raw_dataset_dir / request.dataset_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Dataset manifest not found")
    job_id = f"train_{uuid4().hex}"
    TRAINING_JOBS[job_id] = {"status": "queued", "request": request.dict()}
    background_tasks.add_task(_run_training, job_id, manifest_path)
    return TrainingJobResponse(job_id=job_id, dataset_id=request.dataset_id, status="queued", message="Training job queued")


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str, user_id: str = Depends(require_user_id)):
    if job_id not in TRAINING_JOBS:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TRAINING_JOBS[job_id]
