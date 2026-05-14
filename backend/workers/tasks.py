from pathlib import Path

from backend.ml.training.trainer import create_training_job
from backend.services.dataset_ingestion import DatasetIngestionService
from backend.workers.celery_app import celery_app


@celery_app.task
def ingest_dataset(path: str):
    import asyncio
    return asyncio.run(DatasetIngestionService().ingest_upload(Path(path)))


@celery_app.task
def train_model(manifest_path: str):
    return create_training_job(Path(manifest_path))


@celery_app.task
def run_inference(path: str):
    # Route to inference service in production worker deployment.
    return {"status": "queued_for_inference", "path": path}
