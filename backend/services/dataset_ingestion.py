import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageSequence, UnidentifiedImageError

from backend.core.config import get_settings
from backend.schemas.platform import DatasetCreateResponse, DatasetStats, DatasetStatus

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_ANNOTATION_EXTENSIONS = {".csv", ".json", ".jsonl"}
SUPPORTED_ARCHIVES = {".zip"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf"}


@dataclass
class FileMetadata:
    path: str
    sha256: str
    extension: str
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    is_corrupted: bool = False
    is_duplicate: bool = False
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    annotation: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)


class DatasetIngestionService:
    """Accepts arbitrary user datasets and normalizes them into a trainable manifest.

    This service deliberately avoids assuming one fixed dataset layout. It searches
    recursively for supported image/PDF/annotation files, computes metadata, detects
    corrupted and duplicate files, and emits a canonical manifest used by training,
    indexing, and active learning.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    async def ingest_upload(self, source_path: Path, dataset_name: Optional[str] = None) -> DatasetCreateResponse:
        dataset_id = f"ds_{uuid4().hex}"
        dataset_root = self.settings.raw_dataset_dir / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)

        if source_path.suffix.lower() in SUPPORTED_ARCHIVES:
            self._safe_extract_zip(source_path, dataset_root)
        elif source_path.is_dir():
            self._copy_tree(source_path, dataset_root)
        else:
            target = dataset_root / source_path.name
            shutil.copy2(source_path, target)

        annotations = self._load_annotations(dataset_root)
        metadata, warnings = self._scan_files(dataset_root, annotations)
        stats = self._build_stats(metadata)
        manifest_path = dataset_root / "manifest.json"
        manifest = {
            "dataset_id": dataset_id,
            "name": dataset_name or source_path.stem,
            "root": str(dataset_root),
            "files": [asdict(item) for item in metadata],
            "stats": stats.dict(),
            "warnings": warnings,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        status = DatasetStatus.valid if stats.valid_images > 0 else DatasetStatus.failed
        return DatasetCreateResponse(
            dataset_id=dataset_id,
            status=status,
            stats=stats,
            warnings=warnings,
            metadata_path=str(manifest_path),
        )

    def _safe_extract_zip(self, zip_path: Path, output_dir: Path) -> None:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = output_dir / member.filename
                resolved = target.resolve()
                if not str(resolved).startswith(str(output_dir.resolve())):
                    raise ValueError(f"Unsafe path inside ZIP: {member.filename}")
                archive.extract(member, output_dir)

    def _copy_tree(self, source: Path, target: Path) -> None:
        for path in source.rglob("*"):
            if path.is_file():
                destination = target / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    def _load_annotations(self, root: Path) -> Dict[str, Dict]:
        annotations: Dict[str, Dict] = {}
        for path in root.rglob("*"):
            if path.suffix.lower() not in SUPPORTED_ANNOTATION_EXTENSIONS:
                continue
            if path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        key = row.get("filename") or row.get("image") or row.get("path")
                        if key:
                            annotations[Path(key).name] = row
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("annotations", [])
                for row in rows:
                    key = row.get("filename") or row.get("image") or row.get("path")
                    if key:
                        annotations[Path(key).name] = row
            elif path.suffix.lower() == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    key = row.get("filename") or row.get("image") or row.get("path")
                    if key:
                        annotations[Path(key).name] = row
        return annotations

    def _scan_files(self, root: Path, annotations: Dict[str, Dict]) -> Tuple[List[FileMetadata], List[str]]:
        seen_hashes: Dict[str, str] = {}
        metadata: List[FileMetadata] = []
        warnings: List[str] = []

        for path in root.rglob("*"):
            if not path.is_file() or path.name == "manifest.json":
                continue
            extension = path.suffix.lower()
            if extension not in SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_ANNOTATION_EXTENSIONS:
                warnings.append(f"Unsupported file skipped: {path.name}")
                continue
            if extension in SUPPORTED_ANNOTATION_EXTENSIONS:
                continue

            digest = self._sha256(path)
            duplicate = digest in seen_hashes
            if duplicate:
                warnings.append(f"Duplicate detected: {path.name} matches {seen_hashes[digest]}")
            seen_hashes.setdefault(digest, path.name)

            if extension in SUPPORTED_DOCUMENT_EXTENSIONS:
                metadata.append(FileMetadata(path=str(path), sha256=digest, extension=extension, is_duplicate=duplicate))
                warnings.append(f"PDF detected for conversion pipeline: {path.name}")
                continue

            item = self._inspect_image(path, digest, duplicate, annotations.get(path.name))
            metadata.append(item)
            if item.is_corrupted:
                warnings.append(f"Corrupted image detected: {path.name}")
        return metadata, warnings

    def _inspect_image(self, path: Path, digest: str, duplicate: bool, annotation: Optional[Dict]) -> FileMetadata:
        try:
            with Image.open(path) as image:
                frame = next(ImageSequence.Iterator(image)).convert("RGB")
                width, height = frame.size
                array = np.array(frame)
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
                return FileMetadata(
                    path=str(path),
                    sha256=digest,
                    extension=path.suffix.lower(),
                    width=width,
                    height=height,
                    channels=3,
                    is_duplicate=duplicate,
                    blur_score=blur_score,
                    brightness=brightness,
                    annotation=annotation,
                    warnings=self._quality_warnings(blur_score, brightness),
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return FileMetadata(
                path=str(path),
                sha256=digest,
                extension=path.suffix.lower(),
                is_corrupted=True,
                is_duplicate=duplicate,
                annotation=annotation,
                warnings=[str(exc)],
            )

    def _quality_warnings(self, blur_score: float, brightness: float) -> List[str]:
        warnings: List[str] = []
        if blur_score < 80:
            warnings.append("low_sharpness")
        if brightness < 55:
            warnings.append("low_light")
        if brightness > 220:
            warnings.append("overexposed")
        return warnings

    def _build_stats(self, metadata: Iterable[FileMetadata]) -> DatasetStats:
        items = list(metadata)
        images = [item for item in items if item.extension in SUPPORTED_IMAGE_EXTENSIONS and not item.is_corrupted]
        formats: Dict[str, int] = {}
        resolutions: Dict[str, int] = {}
        for item in items:
            formats[item.extension] = formats.get(item.extension, 0) + 1
            if item.width and item.height:
                bucket = f"{round(item.width / 256) * 256}x{round(item.height / 256) * 256}"
                resolutions[bucket] = resolutions.get(bucket, 0) + 1

        widths = [item.width for item in images if item.width]
        heights = [item.height for item in images if item.height]
        low_light = [item for item in images if item.brightness is not None and item.brightness < 55]
        blurry = [item for item in images if item.blur_score is not None and item.blur_score < 80]
        annotated = [item for item in items if item.annotation]

        return DatasetStats(
            total_files=len(items),
            valid_images=len(images),
            scanned_pdfs=sum(1 for item in items if item.extension == ".pdf"),
            annotations=len(annotated),
            corrupted_files=sum(1 for item in items if item.is_corrupted),
            duplicate_files=sum(1 for item in items if item.is_duplicate),
            formats=formats,
            resolutions=resolutions,
            mean_width=float(np.mean(widths)) if widths else None,
            mean_height=float(np.mean(heights)) if heights else None,
            low_light_ratio=len(low_light) / len(images) if images else None,
            blur_ratio=len(blurry) / len(images) if images else None,
            estimated_script_count=self._estimate_script_count(annotated),
        )

    def _estimate_script_count(self, annotated: List[FileMetadata]) -> Optional[int]:
        labels = set()
        for item in annotated:
            if item.annotation:
                label = item.annotation.get("script") or item.annotation.get("language") or item.annotation.get("label")
                if label:
                    labels.add(label)
        return len(labels) if labels else None

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
