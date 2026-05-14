from celery import Celery

from backend.core.config import get_settings

settings = get_settings()
celery_app = Celery("ancient_ai", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_routes = {
    "backend.workers.tasks.ingest_dataset": {"queue": "ingestion"},
    "backend.workers.tasks.train_model": {"queue": "training"},
    "backend.workers.tasks.run_inference": {"queue": "inference"},
}
