from __future__ import annotations

import cv2
import numpy as np


class LineSegmenter:
    """Line-by-line segmentation using projection profile and contour fallback."""

    def segment(self, ocr_image: np.ndarray, binary_image: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
        gray = ocr_image if len(ocr_image.shape) == 2 else cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
        foreground = 255 - binary_image if np.mean(binary_image) > 127 else binary_image
        projection_boxes = self._projection_lines(foreground)
        boxes = projection_boxes if projection_boxes else self._contour_lines(foreground)

        h, w = gray.shape[:2]
        line_images: list[np.ndarray] = []
        line_meta: list[dict] = []
        for index, (x, y, bw, bh) in enumerate(boxes):
            pad_y = max(4, int(bh * 0.18))
            pad_x = max(6, int(bw * 0.025))
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            if y1 - y0 < 8 or x1 - x0 < 12:
                continue
            line_images.append(gray[y0:y1, x0:x1])
            line_meta.append({"index": index, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
        if not line_images:
            return [gray], [{"index": 0, "x": 0, "y": 0, "w": w, "h": h}]
        return line_images, line_meta

    def _projection_lines(self, foreground: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = foreground.shape[:2]
        projection = np.count_nonzero(foreground > 0, axis=1)
        if projection.max(initial=0) == 0:
            return []
        threshold = max(2, int(0.012 * w))
        active = projection > threshold
        boxes: list[tuple[int, int, int, int]] = []
        start = None
        for y, is_active in enumerate(active):
            if is_active and start is None:
                start = y
            elif not is_active and start is not None:
                self._append_projection_box(boxes, foreground, start, y)
                start = None
        if start is not None:
            self._append_projection_box(boxes, foreground, start, h)
        return self._merge_close_lines(boxes, h)

    def _append_projection_box(self, boxes: list[tuple[int, int, int, int]], foreground: np.ndarray, y0: int, y1: int) -> None:
        if y1 - y0 < 6:
            return
        rows = foreground[y0:y1, :]
        cols = np.where(np.count_nonzero(rows > 0, axis=0) > 0)[0]
        if cols.size == 0:
            return
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    def _merge_close_lines(self, boxes: list[tuple[int, int, int, int]], image_height: int) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        merged = [boxes[0]]
        max_gap = max(3, image_height // 150)
        for x, y, w, h in boxes[1:]:
            px, py, pw, ph = merged[-1]
            if y - (py + ph) <= max_gap:
                x0, y0 = min(px, x), py
                x1, y1 = max(px + pw, x + w), y + h
                merged[-1] = (x0, y0, x1 - x0, y1 - y0)
            else:
                merged.append((x, y, w, h))
        return merged

    def _contour_lines(self, foreground: np.ndarray) -> list[tuple[int, int, int, int]]:
        width_kernel = max(18, foreground.shape[1] // 35)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width_kernel, 3))
        connected = cv2.dilate(foreground, kernel, iterations=1)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        return sorted([box for box in boxes if box[2] * box[3] > 80], key=lambda item: item[1])
