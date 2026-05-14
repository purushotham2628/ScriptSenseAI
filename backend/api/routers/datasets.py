import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from backend.core.config import get_settings
from backend.core.security import require_user_id
from backend.schemas.platform import DatasetCreateResponse
from backend.services.dataset_ingestion import DatasetIngestionService

router = APIRouter()


@router.post("/upload", response_model=DatasetCreateResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(require_user_id),
):
    settings = get_settings()
    suffix = Path(file.filename or "dataset.zip").suffix
    temp_path = settings.storage_root / "tmp" / f"{uuid4().hex}{suffix}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if temp_path.stat().st_size > max_bytes:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Dataset is larger than configured upload limit")

    # Malware scanning hook: call ClamAV/S3 object scanner before ingestion in production.
    response = await DatasetIngestionService().ingest_upload(temp_path, dataset_name=Path(file.filename or "dataset").stem)
    background_tasks.add_task(temp_path.unlink, missing_ok=True)
    return response


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, user_id: str = Depends(require_user_id)):
    settings = get_settings()
    matches = list(settings.raw_dataset_dir.glob(f"{dataset_id}/manifest.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return matches[0].read_text(encoding="utf-8")
