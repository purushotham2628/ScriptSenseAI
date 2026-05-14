from fastapi import APIRouter, Depends

from backend.core.security import require_user_id
from backend.ml.active_learning.service import ActiveLearningService
from backend.schemas.platform import AnnotationCorrection

router = APIRouter()
CORRECTION_BUFFER = []


@router.post("/corrections")
async def submit_correction(correction: AnnotationCorrection, user_id: str = Depends(require_user_id)):
    sample = ActiveLearningService().build_training_sample(correction)
    CORRECTION_BUFFER.append(sample)
    return {"status": "accepted", "buffer_size": len(CORRECTION_BUFFER), "sample": sample}


@router.get("/buffer")
async def get_correction_buffer(user_id: str = Depends(require_user_id)):
    return {"samples": CORRECTION_BUFFER[-100:]}
