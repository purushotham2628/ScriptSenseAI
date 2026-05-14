import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.core.config import get_settings
from backend.core.security import require_user_id
from backend.ml.inference.pipeline import ResearchGradeInferencePipeline
from backend.schemas.platform import PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en"),
    dataset_id: str | None = Form(None),
    user_id: str = Depends(require_user_id),
):
    settings = get_settings()
    suffix = Path(file.filename or "input.png").suffix or ".png"
    input_path = settings.storage_root / "inference" / f"{uuid4().hex}{suffix}"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        return await ResearchGradeInferencePipeline().predict_image(input_path, source_language, target_language, dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
