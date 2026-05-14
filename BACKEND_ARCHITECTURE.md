# Ancient Script AI Platform - Production Backend Architecture

This backend is designed as a startup/research-grade AI platform rather than a fixed-dataset OCR demo. The central design goal is **generalization to outside and unseen datasets**: new scripts, new image qualities, damaged manuscripts, noisy scans, faded characters, mixed symbol systems, and partially broken inscriptions.

## Architecture Diagram

```text
Clients / Frontend / Research UI
        |
        | REST + WebSocket + JWT
        v
FastAPI API Gateway
        |
        +-- Auth Router ---------------------- PostgreSQL.users
        |
        +-- Dataset Router ------------------- Object Storage / storage/datasets
        |       |
        |       +-- ZIP/folder/PDF/CSV ingestion
        |       +-- validation, stats, duplicate detection
        |       +-- canonical manifest.json
        |
        +-- Inference Router
        |       |
        |       v
        |   Multi-stage AI Pipeline
        |       UPLOAD
        |        -> PREPROCESS
        |        -> SEGMENTATION
        |        -> FEATURE EXTRACTION
        |        -> OCR
        |        -> LANGUAGE MODEL CORRECTION
        |        -> TRANSLATION
        |        -> RESULT STORAGE
        |        -> VISUALIZATION
        |
        +-- Training Router ------------------ GPU workers / distributed jobs
        |       |
        |       +-- self-supervised warmup
        |       +-- transfer learning
        |       +-- contrastive / metric learning
        |       +-- domain adaptation
        |       +-- mixed precision training
        |       +-- checkpoint + model registry
        |
        +-- Active Learning Router
        |       |
        |       +-- uncertainty queue
        |       +-- human corrections
        |       +-- replay buffer
        |       +-- continual fine-tuning
        |
        +-- WebSocket Progress Router
                |
                +-- live upload/training/inference progress

Persistence Layer
        |
        +-- PostgreSQL: users, datasets, model metadata, training jobs, predictions, annotations, embeddings metadata
        +-- MongoDB: flexible OCR traces, raw pipeline artifacts, experiment documents
        +-- FAISS/ChromaDB: symbol/script/context embeddings
        +-- Object storage: raw uploads, processed images, checkpoints, ONNX exports
        +-- Redis/Celery/RQ: background jobs and rate limiting
```

## Backend Folder Structure

```text
backend/
  app.py                              # FastAPI app, CORS, static serving, compatibility /process
  api/
    router.py                         # API router composition
    routers/
      auth.py                         # JWT auth templates
      datasets.py                     # ZIP/folder/image dataset upload endpoints
      inference.py                    # research-grade prediction endpoint
      training.py                     # async training job endpoint
      active_learning.py              # human correction + continual learning buffer
      websocket.py                    # live progress updates
  core/
    config.py                         # environment-driven settings
    logging.py                        # structured logging bootstrap
    security.py                       # JWT, password hashing, dependency hooks
  db/
    models.py                         # PostgreSQL schema models
    session.py                        # async SQLAlchemy session
  schemas/
    platform.py                       # Pydantic API contracts
  services/
    dataset_ingestion.py              # dynamic dataset parser/validator/statistics
  ml/
    preprocessing/
      advanced_preprocessor.py        # OpenCV/Pillow robust preprocessing
      augmentation.py                 # Albumentations degradation simulation
    embeddings/
      vector_store.py                 # FAISS/fallback vector index
    models/
      hybrid_ocr.py                   # hybrid ViT/Swin/ConvNeXt-style OCR scaffold
    inference/
      pipeline.py                     # multi-stage inference orchestration
    training/
      objectives.py                   # SSL, contrastive, domain adaptation losses
      trainer.py                      # AMP/checkpoint/TensorBoard training scaffold
    active_learning/
      service.py                      # uncertainty scoring and replay buffer
infra/
  k8s/                                # Kubernetes manifests
Dockerfile
Dockerfile.gpu
docker-compose.yml
.github/workflows/backend-ci.yml
```

## API Endpoints

### Public / compatibility

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves existing frontend |
| `GET` | `/api` | Platform metadata |
| `GET` | `/health` | Health check |
| `POST` | `/process` | Compatibility route for current frontend |

### Production API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Register user template |
| `POST` | `/api/v1/auth/token` | Issue JWT |
| `POST` | `/api/v1/datasets/upload` | Upload ZIP/image/folder archive dataset |
| `GET` | `/api/v1/datasets/{dataset_id}` | Fetch canonical manifest |
| `POST` | `/api/v1/inference/predict` | Run robust unseen-data inference |
| `POST` | `/api/v1/training/jobs` | Start training/fine-tuning job |
| `GET` | `/api/v1/training/jobs/{job_id}` | Poll training job |
| `POST` | `/api/v1/active-learning/corrections` | Submit human correction |
| `GET` | `/api/v1/active-learning/buffer` | Inspect correction replay buffer |
| `WS` | `/api/v1/ws/progress/{job_id}` | Live progress stream |

## Database Schema

### PostgreSQL tables

- `users`: SaaS identity, role, password hash.
- `datasets`: uploaded dataset metadata, status, stats, label map, manifest path.
- `model_versions`: checkpoint registry, metrics, parent model relationship, active version.
- `training_jobs`: job status, strategy, config, logs, metrics.
- `predictions`: input URI, OCR text, corrected text, translation, confidence, anomaly score.
- `annotations`: human labels, corrected symbols, quality score, source.
- `embeddings`: vector index ID, symbol/script label, metadata.

### MongoDB collections

Recommended future collections:

- `pipeline_traces`: full per-stage debug payloads.
- `ocr_regions`: bounding boxes and per-symbol candidates.
- `experiment_configs`: flexible research configs.
- `active_learning_events`: queue state and reviewer actions.

## ML Pipeline

```text
Input image/PDF/page
  -> corruption and format validation
  -> illumination normalization
  -> CLAHE contrast enhancement
  -> denoising
  -> skew correction
  -> stain removal
  -> faded text enhancement
  -> edge sharpening
  -> adaptive thresholding
  -> erosion repair
  -> segmentation
  -> vision encoder embedding
  -> nearest symbol/script retrieval
  -> OCR decoder
  -> confidence calibration + anomaly detection
  -> contextual correction
  -> translation hook
  -> review decision
```

## Recommended Models

### Vision encoder

Use one of:

- `Swin Transformer / SwinV2`: strong for multi-scale texture and visual structure.
- `ConvNeXtV2`: robust convolutional inductive bias plus modern training behavior.
- `ViT/BEiT/DeiT`: best when self-supervised pretraining data is large.

### OCR sequence model

Use one of:

- `TrOCR`: strong pretrained encoder-decoder OCR baseline.
- `CRNN + CTC`: fast, reliable for production low-latency recognition.
- `PaddleOCR`: production-grade detector/recognizer integration baseline.
- `Transformer decoder`: better for complex context and mixed symbol structures.

### Language/context layer

Use:

- HuggingFace seq2seq correction model for OCR denoising.
- Character-level transformer for missing symbol reconstruction.
- GPT-style contextual corrector for ambiguous fragments.
- Domain lexicons and inscription grammar constraints when available.

### Translation

Use:

- NLLB/M2M100 for multilingual translation.
- Custom glossary/lexicon retrieval for ancient scripts.
- Human-verified parallel corpora for script-specific fine-tuning.

## Why This Improves Unseen Dataset Performance

### Dynamic ingestion avoids fixed-dataset assumptions

The ingestion service recursively parses arbitrary uploaded layouts, finds supported images/PDFs/annotations, generates a manifest, computes quality statistics, and builds label maps dynamically. This means new datasets do not need to imitate the original training dataset structure.

### Robust preprocessing reduces domain shift

Unseen manuscripts often differ more by acquisition quality than by script alone. Illumination normalization, denoising, CLAHE, skew correction, stain removal, faded-text enhancement, edge sharpening, and erosion repair reduce nuisance variation before the model sees the image. Less visual domain shift means better generalization.

### Synthetic degradation makes small datasets useful

The augmentation pipeline simulates blur, cracks, occlusion, ink fading, perspective changes, noise, low light, and character degradation. This teaches the model invariances that are common in real archaeological data but rare in small curated datasets.

### Transfer learning gives strong visual priors

A Swin/ViT/ConvNeXt encoder pretrained on large image corpora already understands edges, texture, layout, and shape. Fine-tuning it on manuscript data requires far fewer labels than training from scratch.

### Self-supervised learning uses unlabeled manuscripts

Masked patch reconstruction and contrastive learning let the system learn script-specific visual structure without labels. This is crucial because ancient script datasets are often tiny, expensive, or partially annotated.

### Few-shot learning supports new scripts quickly

Few-shot episodes train the model to compare new symbols by similarity rather than memorize a closed class list. This lets the platform adapt from a handful of corrected examples.

### Contrastive and metric learning provide retrieval fallback

Even when the OCR classifier does not know a symbol, embeddings can retrieve visually similar symbols from FAISS/ChromaDB. The system can return nearest-symbol candidates and uncertainty instead of failing silently.

### Domain adaptation aligns old and new datasets

Domain adaptation losses reduce embedding distribution mismatch between source training data and newly uploaded target data. This improves behavior on new scanners, lighting conditions, parchment textures, and writing styles.

### Unknown-symbol handling prevents forced wrong answers

Dynamic label mapping and unknown-symbol ratios allow the model to admit uncertainty. This is safer for unseen scripts than forcing every glyph into known Latin/Greek classes.

### Active learning closes the loop

Low-confidence, anomalous, or unknown-heavy predictions are flagged for human correction. Corrected samples enter a replay buffer and become training data for continual learning.

### Continual learning avoids full retraining

The replay buffer mixes new corrected samples with older exemplars. This allows incremental fine-tuning on new scripts while reducing catastrophic forgetting of previous scripts.

## Training Pipeline

```text
Dataset manifest
  -> split train/val/test by manuscript/source
  -> dynamic label mapping
  -> self-supervised warmup on all images
  -> supervised OCR fine-tuning on labeled samples
  -> contrastive metric learning on symbol crops
  -> domain adaptation against target dataset
  -> validation with CER/WER/SER/anomaly metrics
  -> early stopping
  -> checkpoint registry
  -> ONNX export
  -> canary deployment
```

## Evaluation Metrics

Use multiple metrics because OCR accuracy alone is not enough for unseen scripts:

- `CER`: character error rate.
- `WER`: word error rate where word boundaries exist.
- `SER`: symbol error rate for glyph-level scripts.
- `Top-k symbol retrieval accuracy`: whether the correct symbol appears in nearest candidates.
- `Unknown-symbol calibration`: whether unknowns are flagged instead of hallucinated.
- `Expected calibration error`: confidence reliability.
- `Anomaly AUROC`: ability to detect unseen scripts or out-of-distribution pages.
- `Latency p50/p95/p99`: production inference performance.
- `Review yield`: percentage of active-learning reviews that improve training.

## Deployment Pipeline

```text
Pull request
  -> lint + type checks + unit tests
  -> build CPU Docker image
  -> optional GPU image
  -> run API smoke tests
  -> push image to registry
  -> deploy to staging Kubernetes
  -> run model canary evaluation
  -> promote to production
```

## Tech Stack Justification

- `FastAPI`: async REST/WebSocket APIs, strong schema generation, production-friendly.
- `PostgreSQL`: relational metadata, users, jobs, predictions, model versions.
- `MongoDB`: flexible research traces and OCR region artifacts.
- `FAISS/ChromaDB`: scalable similarity search for symbol/script embeddings.
- `Redis + Celery/RQ`: durable async training and ingestion queues.
- `OpenCV + Pillow`: low-level image validation and restoration.
- `Albumentations`: strong augmentation library for visual robustness.
- `PyTorch`: research velocity and production model export.
- `HuggingFace/timm`: access to modern pretrained encoders and language models.
- `TensorBoard/W&B`: experiment tracking.
- `ONNX Runtime/TensorRT`: inference optimization.
- `Docker/Kubernetes`: reproducible deployment and horizontal scaling.

## Step-by-Step Implementation Plan

1. Keep the current frontend and `/process` compatibility route.
2. Move all new clients to `/api/v1/*` endpoints with JWT.
3. Add real PostgreSQL connection and run migrations with Alembic.
4. Replace in-memory training job dictionaries with Celery/RQ workers.
5. Add S3/MinIO object storage for datasets and checkpoints.
6. Integrate ClamAV or object-storage malware scanning for uploads.
7. Add real FAISS persistence or ChromaDB service.
8. Swap the lightweight model scaffold for Swin/ConvNeXt/TrOCR/PaddleOCR.
9. Build DataLoaders from `manifest.json`.
10. Add self-supervised pretraining on unlabeled uploaded pages.
11. Add symbol crop generation and contrastive training.
12. Add active-learning review UI endpoint integration.
13. Add model registry promotion rules and ONNX export.
14. Add Kubernetes GPU worker deployments.
15. Add canary evaluation before model promotion.

## Future Scalability Plan

- Use presigned uploads to object storage for multi-GB datasets.
- Store page-level and crop-level embeddings separately.
- Shard FAISS indexes by script family or dataset domain.
- Use GPU inference replicas with ONNX/TensorRT.
- Use separate CPU preprocessing workers and GPU OCR workers.
- Add model ensembles for detector/recognizer/context correction.
- Add per-tenant model adapters with LoRA for SaaS customers.
- Add weak supervision from unlabeled manuscript collections.
- Add reviewer consensus and annotation quality scoring.
- Add model cards and dataset lineage for research reproducibility.
