"""
PP-OCRv6 pure ONNXRuntime API inference implementation.

Does not depend on PaddleX; uses only onnxruntime + opencv + numpy + pyclipper.
Following the PaddleX source, fully reproduces the detection + recognition pre-/post-processing logic.

Usage::

    from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

    with PPOCRv6Onnx("det.onnx", "rec.onnx", "dict.txt") as ocr:
        results = ocr(image_bgr)
        for r in results:
            print(r.text, r.score)
"""

from __future__ import annotations

import copy
import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

__all__ = ["PPOCRv6Onnx", "OCRResult"]

logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent

DEFAULT_PPOCRV6_DICT = str(root / "ppocrv6_onnx_data" / "ppocrv6_dict.txt")

# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OCRResult:
    """A single OCR recognition result."""

    text: str
    """Recognized text."""
    score: float
    """Confidence score in the range [0, 1]."""
    box: List[List[int]]
    """Four vertex coordinates of the detection box [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]"""


# ---------------------------------------------------------------------------
# Automatic ORT provider selection
# ---------------------------------------------------------------------------

def _resolve_ort_providers(prefer_cpu: bool = False) -> List[str]:
    """Return the list of ONNX Runtime execution providers.

    Args:
        prefer_cpu: If True, force CPU (guarantees cross-platform bit-exact precision).
                Otherwise auto-select in the order CoreML (macOS) > CUDA > CPU.
    """
    if prefer_cpu:
        logger.info("ONNX Runtime provider: CPUExecutionProvider (forced)")
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    for preferred in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
        if preferred in available:
            logger.info("ONNX Runtime provider: %s", preferred)
            return [preferred]
    logger.info("ONNX Runtime provider: CPUExecutionProvider (fallback)")
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Shared utility: vertex ordering for the minimum-area bounding rectangle
# ---------------------------------------------------------------------------

def _order_minarea_box_points(
    contour: np.ndarray,
) -> Tuple[List[np.ndarray], float]:
    """Compute the minimum-area bounding rectangle of a contour; return the 4 ordered vertices and the short-side length.

    Vertex order: [top-left, top-right, bottom-right, bottom-left]

    Args:
        contour: OpenCV-style contour points.

    Returns:
        (ordered_points, min_side_length): List of four points and the short-side length of the minAreaRect.
    """
    rrect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rrect), key=lambda p: p[0])

    tl, bl = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    tr, br = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)

    return [pts[tl], pts[tr], pts[br], pts[bl]], min(rrect[1])


# ============================================================
# Detection model preprocessing
# ============================================================

class DetPreProcess:
    """Detection model preprocessing pipeline.

    Executes in order: DetResizeForTest → NormalizeImage → ToCHW → ToBatch.
    """

    _VALID_LIMIT_TYPES = frozenset({"min", "max"})

    def __init__(
        self,
        limit_side_len: int = 64,
        limit_type: str = "min",
        max_side_limit: int = 4000,
    ) -> None:
        if limit_type not in self._VALID_LIMIT_TYPES:
            raise ValueError(
                f"limit_type must be one of {sorted(self._VALID_LIMIT_TYPES)}, "
                f"got {limit_type!r}"
            )
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_limit = max_side_limit
        # ImageNet normalization parameters (BGR order)
        self._scale = np.float32(1.0 / 255.0)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"DetPreProcess(limit={self.limit_side_len}, "
            f"type={self.limit_type}, max_side={self.max_side_limit})"
        )

    def _resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """DetResizeForTest.resize_image_type0.

        Returns:
            (resized_img, ratio_h, ratio_w)
        """
        h, w = img.shape[:2]
        limit = self.limit_side_len

        if self.limit_type == "max":
            ratio = float(limit) / max(h, w) if max(h, w) > limit else 1.0
        else:  # "min"
            ratio = float(limit) / min(h, w) if min(h, w) < limit else 1.0

        rh = int(h * ratio)
        rw = int(w * ratio)

        if max(rh, rw) > self.max_side_limit:
            ratio = float(self.max_side_limit) / max(rh, rw)
            rh, rw = int(rh * ratio), int(rw * ratio)

        rh = max(int(round(rh / 32) * 32), 32)
        rw = max(int(round(rw / 32) * 32), 32)

        if rh == h and rw == w:
            return img, 1.0, 1.0

        resized = cv2.resize(img, (rw, rh))
        return resized, rh / float(h), rw / float(w)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """ImageNet normalization: (img * scale - mean) / std."""
        return (img.astype(np.float32, copy=False) * self._scale - self._mean) / self._std

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full detection preprocessing.

        Args:
            img: BGR image (H, W, 3).

        Returns:
            (tensor, img_shape) — tensor has shape [1, 3, H', W'],
            img_shape = [src_h, src_w, ratio_h, ratio_w].
        """
        src_h, src_w = img.shape[:2]
        resized, ratio_h, ratio_w = self._resize_image(img)
        norm = self._normalize(resized)
        chw = np.transpose(norm, (2, 0, 1))
        tensor = chw[np.newaxis, ...].astype(np.float32, copy=False)
        shape = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)
        return tensor, shape


# ============================================================
# Detection model postprocessing (DBPostProcess)
# ============================================================

class DBPostProcess:
    """DB postprocessing: extract quadrilateral text boxes from the probability map (box_type="quad")."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        min_size: int = 3,
    ) -> None:
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_size = min_size

    def __repr__(self) -> str:
        return (
            f"DBPostProcess(thresh={self.thresh}, box_thresh={self.box_thresh}, "
            f"unclip={self.unclip_ratio})"
        )

    @staticmethod
    def _box_score(bitmap: np.ndarray, points: np.ndarray) -> float:
        """Compute the mean probability inside a polygon box (fast mode).

        Args:
            bitmap: Probability map (H, W).
            points: Vertex coordinates (4, 2).

        Returns:
            Mean value inside the box.
        """
        h, w = bitmap.shape[:2]
        box = points.astype(np.float32, copy=True)

        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        """Expand the polygon by unclip_ratio using pyclipper.

        Args:
            box: Vertices (N, 2), float32 or int32.
                **Note: the original dtype must be preserved; do not force-convert to int32,
                because pyclipper uses sub-pixel precision for float32.**

        Returns:
            The expanded polygon.
        """
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, closed=True)
        distance = area * self.unclip_ratio / length

        po = pyclipper.PyclipperOffset()
        po.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            expanded = np.asarray(po.Execute(distance))
        except pyclipper.PyclipperException:
            paths = po.Execute(distance)
            if not paths:
                return box
            expanded = np.asarray(paths[0])
        return expanded

    def _extract_boxes(
        self,
        prob: np.ndarray,
        bitmap: np.ndarray,
        dst_w: int,
        dst_h: int,
    ) -> Tuple[np.ndarray, List[float]]:
        """Extract text boxes from the binarized bitmap and map them back to original image coordinates.

        Args:
            prob: Probability map (H, W).
            bitmap: Binarized mask (H, W).
            dst_w: Original image width.
            dst_h: Original image height.

        Returns:
            (boxes, scores): boxes is (N, 4, 2) int16, scores is a list of N floats.
        """
        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # Compatible with opencv 3 / 4 / 5 return values
        n_outs = len(outs)
        if n_outs == 3:
            contours = outs[1]
        elif n_outs == 2:
            contours = outs[0]
        else:
            raise RuntimeError(f"cv2.findContours returned {n_outs} values")

        ws, hs = dst_w / bitmap.shape[1], dst_h / bitmap.shape[0]
        boxes: List[np.ndarray] = []
        scores: List[float] = []

        for contour in contours[:self.max_candidates]:
            pts, sside = _order_minarea_box_points(contour)
            if sside < self.min_size:
                continue

            pts_np = np.array(pts, dtype=np.float32)
            score = self._box_score(prob, pts_np.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            expanded = self._unclip(pts_np).reshape(-1, 1, 2)
            expanded_pts, sside2 = _order_minarea_box_points(expanded)
            if sside2 < self.min_size + 2:
                continue

            box = np.array(expanded_pts, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] * ws), 0, dst_w)
            box[:, 1] = np.clip(np.round(box[:, 1] * hs), 0, dst_h)

            boxes.append(box.astype(np.int16))
            scores.append(score)

        if not boxes:
            return np.empty((0, 4, 2), dtype=np.int16), []
        return np.stack(boxes, axis=0), scores

    def __call__(
        self, pred: np.ndarray, img_shape: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """Run DB postprocessing.

        Args:
            pred: Model output (1, 1, H', W').
            img_shape: [src_h, src_w, ratio_h, ratio_w].

        Returns:
            (boxes, scores)
        """
        prob_map = pred[0, 0, :, :]
        segmentation = prob_map > self.thresh
        return self._extract_boxes(prob_map, segmentation, int(img_shape[1]), int(img_shape[0]))


# ============================================================
# Box sorting
# ============================================================

def sort_quad_boxes(boxes: np.ndarray) -> np.ndarray:
    """Sort text boxes in reading order (top to bottom, left to right).

    Algorithm: sort primarily by (y0, x0), then for adjacent boxes whose y0 difference is < 10 (treated as the same line),
    stably re-order them by x0 using insertion sort. Time complexity is O(n²), acceptable for a typical few dozen boxes.

    Args:
        boxes: (N, 4, 2) int16 array of text boxes.

    Returns:
        Sorted array of the same shape.
    """
    if len(boxes) <= 1:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    items = list(sorted_boxes)
    n = len(items)
    for i in range(n - 1):
        for j in range(i, -1, -1):
            if abs(items[j + 1][0][1] - items[j][0][1]) < 10 and (
                items[j + 1][0][0] < items[j][0][0]
            ):
                items[j], items[j + 1] = items[j + 1], items[j]
            else:
                break
    return np.array(items, dtype=boxes.dtype)


# ============================================================
# Text region cropping
# ============================================================

def _rotate_crop_image(
    img: np.ndarray, points: np.ndarray
) -> Optional[np.ndarray]:
    """Crop a rotated text region via perspective transform.

    Args:
        img: Source image (H, W, 3) uint8.
        points: Four vertices (4, 2), in the order [top-left, top-right, bottom-right, bottom-left].

    Returns:
        The cropped image, or None (invalid arguments / perspective transform failed).
    """
    pts = points.astype(np.float32)

    crop_w = int(max(
        np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])
    ))
    crop_h = int(max(
        np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])
    ))
    if crop_w < 1 or crop_h < 1:
        return None

    dst_pts = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])

    try:
        M = cv2.getPerspectiveTransform(pts, dst_pts)
    except cv2.error:
        logger.debug("Degenerate quadrilateral in perspective crop, skipping")
        return None

    cropped = cv2.warpPerspective(
        img, M, (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if cropped.shape[0] / cropped.shape[1] >= 1.5:
        cropped = np.rot90(cropped)
    return cropped


def _minarea_rect_crop(img: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
    """Get the minimum-area bounding rectangle and crop it with a perspective transform.

    Args:
        img: Source image.
        poly: Quadrilateral vertices (4, 2).

    Returns:
        The cropped result, or None.
    """
    box, _ = _order_minarea_box_points(poly.astype(np.int32))
    return _rotate_crop_image(img, np.array(box))


def crop_by_polys(
    img: np.ndarray, dt_polys: np.ndarray
) -> List[Optional[np.ndarray]]:
    """Batch-crop text regions according to the detection boxes.

    Args:
        img: Source image (H, W, 3) uint8.
        dt_polys: (N, 4, 2) array of quadrilaterals.

    Returns:
        List of crop results with the same length as the input; failed entries are None.
        The caller is responsible for filtering them by index to keep alignment.
    """
    results: List[Optional[np.ndarray]] = []
    for poly in dt_polys:
        crop = _minarea_rect_crop(img, copy.deepcopy(poly))
        if crop is not None and crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            results.append(crop)
        else:
            results.append(None)
    return results


# ============================================================
# Recognition model preprocessing
# ============================================================

class RecPreProcess:
    """Recognition model preprocessing: OCRReisizeNormImg + ToBatch.

    Fixed height of 48; width is computed dynamically from the aspect ratio (lower bound 320, upper bound max_width, default 3200),
    normalized to [-1, 1], with zero padding on the right to align within a batch.
    """

    def __init__(self, rec_image_shape: Tuple[int, int, int] = (3, 48, 320),
                 width_buckets: Optional[Sequence[int]] = None,
                 max_width: int = 3200) -> None:
        self._c, self._h, self._w_min = rec_image_shape
        if max_width < self._w_min:
            raise ValueError(
                f"max_width ({max_width}) must be >= the minimum width ({self._w_min})")

        self._buckets = sorted(set(int(b) for b in width_buckets)) if width_buckets else None
        self._max_w = int(max_width)
        # 256-entry float32 lookup table: identical, element by element, to the (x/255 - 0.5)/0.5
        # float32 operation sequence in _resize_norm, so results are bit-for-bit identical, but only one lookup is needed instead of multiple floating-point passes.
        _x = np.arange(256, dtype=np.float32)
        _x *= 1.0 / 255.0
        self._lut = (_x - 0.5) / 0.5

    def __repr__(self) -> str:
        return f"RecPreProcess(C={self._c}, H={self._h}, W_min={self._w_min})"

    def _resize_norm(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """Resize and normalize to [-1, 1], with zero padding on the right.

        Args:
            img: Input image (H, W, 3) uint8.
            max_wh_ratio: Target maximum aspect ratio.

        Returns:
            Preprocessed tensor (C, H, target_W) float32.
        """
        target_w = int(self._h * max_wh_ratio)

        if target_w > self._max_w:
            resized = cv2.resize(img, (self._max_w, self._h))
            actual_w = self._max_w
            target_w = self._max_w
        else:
            h, w = img.shape[:2]
            actual_w = min(int(math.ceil(self._h * w / float(h))), target_w)
            resized = cv2.resize(img, (actual_w, self._h))

        # CHW → normalize to [-1, 1]
        chw = np.transpose(resized.astype(np.float32, copy=False), (2, 0, 1))
        chw *= 1.0 / 255.0
        chw = (chw - 0.5) / 0.5

        padded = np.zeros((self._c, self._h, target_w), dtype=np.float32)
        padded[:, :, :actual_w] = chw
        return padded

    def _resize_single(self, img: np.ndarray) -> np.ndarray:
        max_ratio = max(self._w_min / self._h, img.shape[1] / float(img.shape[0]))
        return self._resize_norm(img, max_ratio)

    def __call__(self, imgs: List[np.ndarray]) -> np.ndarray:
        """Batch preprocessing.

        The output is bit-for-bit identical to calling ``_resize_single`` per image and then padding + stacking,
        but the output tensor is allocated only once for the whole batch (np.zeros is lazily zeroed, so the padded region is never touched),
        and each image is written directly into its slice, saving the per-image full-width canvas, np.pad, np.stack and astype copies.

        Args:
            imgs: List of BGR images.

        Returns:
            Batch tensor (N, C, H, W_max) float32.
        """
        h = self._h
        plan: List[int] = []
        batch_w = 0
        for img in imgs:
            ih, iw = img.shape[:2]
            max_ratio = max(self._w_min / h, iw / float(ih))
            target_w = int(h * max_ratio)
            if target_w > self._max_w:
                actual_w = target_w = self._max_w
            else:
                actual_w = min(int(math.ceil(h * iw / float(ih))), target_w)
            plan.append(actual_w)
            if target_w > batch_w:
                batch_w = target_w

        if self._buckets:
            batch_w = next((b for b in self._buckets if b >= batch_w), batch_w)

        out = np.zeros((len(imgs), self._c, h, batch_w), dtype=np.float32)
        lut = self._lut
        for i, (img, actual_w) in enumerate(zip(imgs, plan)):
            resized = cv2.resize(img, (actual_w, h))
            if resized.dtype == np.uint8:
                # (h, w, 3) uint8 -> float32 normalized to [-1, 1], then transposed to CHW and written in
                out[i, :, :, :actual_w] = cv2.LUT(resized, lut).transpose(2, 0, 1)
            else:  # non-uint8 input: keep the original float path, behavior unchanged
                chw = np.transpose(resized.astype(np.float32, copy=False), (2, 0, 1))
                chw = chw * (1.0 / 255.0)
                out[i, :, :, :actual_w] = (chw - 0.5) / 0.5
        return out


# ============================================================
# Recognition model postprocessing (CTCLabelDecode)
# ============================================================

class CTCLabelDecode:
    """CTC greedy decode: argmax → collapse consecutive repeats → remove blank → look up characters."""

    def __init__(self, character_dict_path: str) -> None:
        self._blank = 0
        if character_dict_path is None:
            character_dict_path = DEFAULT_PPOCRV6_DICT
        with open(character_dict_path, encoding="utf-8") as f:
            raw = tuple(line.rstrip("\n\r") for line in f)
        self._chars: Tuple[str, ...] = ("blank", *raw)

    def __repr__(self) -> str:
        return f"CTCLabelDecode(vocab_size={len(self._chars)})"

    def __len__(self) -> int:
        return len(self._chars)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (including blank)."""
        return len(self._chars)

    def decode(
        self, indices: np.ndarray, probs: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        """Run CTC greedy decode on a batch of index sequences.

        Args:
            indices: (batch, seq_len) int64 indices.
            probs:  (batch, seq_len) float32 per-frame probabilities (optional, used to compute confidence).

        Returns:
            [(text, confidence), ...] with the same length as the batch.
        """
        # Compute the "collapse consecutive repeats" and "remove blank" masks for the whole batch at once
        keep = np.ones(indices.shape, dtype=bool)
        keep[:, 1:] = indices[:, 1:] != indices[:, :-1]
        keep &= indices != self._blank

        results: List[Tuple[str, float]] = []
        chars = self._chars
        for b in range(len(indices)):
            k = keep[b]
            text = "".join(chars[idx] for idx in indices[b][k])
            if probs is None:
                score = 1.0
            elif k.any():
                score = float(probs[b][k].mean())
            else:
                score = 0.0
            results.append((text, score))
        return results

    def __call__(self, model_output: np.ndarray) -> Tuple[List[str], List[float]]:
        """Run CTC decode on the model output.

        Args:
            model_output: (batch, seq_len, num_classes) float32.

        Returns:
            (texts, scores)
        """
        output = np.asarray(model_output)
        indices = output.argmax(axis=-1)
        # Take the max directly at the argmax positions, equivalent to output.max(axis=-1),
        # but avoids scanning the large (batch, seq_len, num_classes) tensor again
        probs = np.take_along_axis(output, indices[..., None], axis=-1)[..., 0]
        decoded = self.decode(indices, probs)
        return [d[0] for d in decoded], [d[1] for d in decoded]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _require_file(path: str, name: str) -> None:
    """Verify that the file exists, otherwise raise a clear FileNotFoundError."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name}: file not found: {path}")


# ============================================================
# Main inference class
# ============================================================

class PPOCRv6Onnx:
    """PP-OCRv6 pure ONNX Runtime inferencer.

    Uses the CPU provider by default to guarantee cross-platform bit-exact precision.
    For GPU / CoreML acceleration, pass ``prefer_accelerator=True``.

    Typical usage::

        with PPOCRv6Onnx("det.onnx", "rec.onnx", "dict.txt") as ocr:
            results = ocr(image_bgr)
            for r in results:
                print(r.text, r.score, r.box)

    Manual close is also supported::

        ocr = PPOCRv6Onnx(...)
        try:
            results = ocr(img)
        finally:
            ocr.close()
    """

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        rec_char_dict_path: str,
        *,
        det_limit_side_len: int = 64,
        det_limit_type: str = "min",
        det_max_side_limit: int = 4000,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        det_unclip_ratio: float = 1.5,
        rec_image_shape: Tuple[int, int, int] = (3, 48, 320),
        rec_batch_size: int = 6,
        prefer_accelerator: bool = False,
    ) -> None:
        # ── Argument validation ──
        _require_file(det_model_path, "det_model_path")
        _require_file(rec_model_path, "rec_model_path")

        if rec_char_dict_path is None:
            rec_char_dict_path = DEFAULT_PPOCRV6_DICT
        _require_file(rec_char_dict_path, "rec_char_dict_path")
        if rec_batch_size < 1:
            raise ValueError(f"rec_batch_size must be >= 1, got {rec_batch_size}")

        # Default to CPU for bit-exact cross-platform precision; accelerators must be explicitly opted in
        providers = _resolve_ort_providers(prefer_cpu=not prefer_accelerator)

        self._det_session = ort.InferenceSession(det_model_path, providers=providers)
        self._rec_session = ort.InferenceSession(rec_model_path, providers=providers)
        self._det_input = self._det_session.get_inputs()[0].name
        self._rec_input = self._rec_session.get_inputs()[0].name

        self._det_pre = DetPreProcess(
            limit_side_len=det_limit_side_len,
            limit_type=det_limit_type,
            max_side_limit=det_max_side_limit,
        )
        self._det_post = DBPostProcess(
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
        )
        self._rec_pre = RecPreProcess(rec_image_shape=rec_image_shape)
        self._rec_post = CTCLabelDecode(rec_char_dict_path)
        self._rec_bs = rec_batch_size

        self._closed = False

    # ── Internal decorators ──

    @staticmethod
    def _require_open(method: Callable) -> Callable:
        """Decorator: ensure the session is not closed before allowing inference methods to be called."""

        @functools.wraps(method)
        def wrapper(self: PPOCRv6Onnx, *args: Any, **kwargs: Any) -> Any:
            if self._closed:
                raise RuntimeError(
                    "PPOCRv6Onnx has been closed and can no longer be used."
                )
            return method(self, *args, **kwargs)

        return wrapper

    # ── Resource management ──

    def close(self) -> None:
        """Release ONNX Runtime session resources. Idempotent."""
        if not self._closed:
            # InferenceSession has no explicit close method; just drop the references
            self._det_session = None  # type: ignore[assignment]
            self._rec_session = None  # type: ignore[assignment]
            self._closed = True

    def __enter__(self) -> PPOCRv6Onnx:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PPOCRv6Onnx(det={self._det_pre!r}, rec={self._rec_pre!r}, "
            f"vocab={self._rec_post.vocab_size}, closed={self._closed})"
        )

    # ── Inference ──

    @_require_open
    def detect(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Text detection.

        Args:
            img_bgr: BGR image (H, W, 3).

        Returns:
            (boxes, scores): boxes has shape (N, 4, 2) int16, scores has length N.
        """
        x, shape = self._det_pre(img_bgr)
        outputs = self._det_session.run(None, {self._det_input: x})
        return self._det_post(outputs[0], shape)

    @_require_open
    def recognize(self, img_list: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """Text recognition (batch supported).

        Args:
            img_list: List of cropped BGR text-line images.

        Returns:
            (texts, scores): same length as the input.
        """
        n = len(img_list)
        if n == 0:
            return [], []

        # Sort by aspect ratio to optimize batch padding; restore the original order after inference
        indexed = sorted(
            [(i, img.shape[1] / float(img.shape[0])) for i, img in enumerate(img_list)],
            key=lambda x: x[1],
        )
        order = [idx for idx, _ in indexed]
        sorted_imgs = [img_list[idx] for idx in order]

        texts: List[Optional[str]] = [None] * n
        scores: List[Optional[float]] = [None] * n

        for start in range(0, n, self._rec_bs):
            batch = sorted_imgs[start:start + self._rec_bs]
            x = self._rec_pre(batch)
            outputs = self._rec_session.run(None, {self._rec_input: x})
            t, s = self._rec_post(outputs[0])
            for j, (text, score) in enumerate(zip(t, s)):
                orig = order[start + j]
                texts[orig] = text
                scores[orig] = score

        return (
            [t if t is not None else "" for t in texts],
            [s if s is not None else 0.0 for s in scores],
        )

    @_require_open
    def __call__(self, img_bgr: np.ndarray) -> List[OCRResult]:
        """Full OCR pipeline: detection → sorting → cropping → recognition.

        Args:
            img_bgr: BGR image (H, W, 3).

        Returns:
            List of OCRResult (in reading order).
        """
        # 1. Detection
        boxes, _ = self.detect(img_bgr)
        if len(boxes) == 0:
            return []

        # 2. Sorting
        sorted_boxes = sort_quad_boxes(boxes)

        # 3. Cropping (failed entries are None, to keep indices aligned)
        crops = crop_by_polys(img_bgr, sorted_boxes)

        # 4. Alignment filtering
        valid_boxes: List[np.ndarray] = []
        valid_crops: List[np.ndarray] = []
        for box, crop in zip(sorted_boxes, crops):
            if crop is not None:
                valid_boxes.append(box)
                valid_crops.append(crop)

        if not valid_crops:
            return []

        # 5. Recognition
        texts, scores = self.recognize(valid_crops)

        return [
            OCRResult(text=text, score=score, box=box.tolist())
            for text, score, box in zip(texts, scores, valid_boxes)
        ]

    # ============================================================
# CLI
# ============================================================

def _main(argv: Optional[List[str]] = None) -> None:
    import argparse
    import sys

    root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="PP-OCRv6 ONNX Runtime inference")
    ap.add_argument(
        "image", nargs="?", default=None,
        help="input image path (default: built-in demo)",
    )
    ap.add_argument(
        "--det-model", default=str(root / "models" / "PP-OCRv6_tiny_det_onnx" / "inference.onnx"),
        help="detection ONNX model path",
    )
    ap.add_argument(
        "--rec-model", default=str(root / "models" / "PP-OCRv6_tiny_rec_0515_onnx" / "inference.onnx"),
        help="recognition ONNX model path",
    )
    ap.add_argument(
        "--dict", default=str(root / "models" / "ppocrv6_dict.txt"),
        help="character dictionary path",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="enable debug logging",
    )
    args = ap.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    image_path = args.image or str(root / "general_ocr_002.png")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read image {image_path}", file=sys.stderr)
        sys.exit(1)

    with PPOCRv6Onnx(args.det_model, args.rec_model, args.dict) as ocr:
        results = ocr(img)

    print(f"\n检测到 {len(results)} 个文本区域：\n")
    for i, r in enumerate(results):
        print(f"  [{i+1:2d}] text=\"{r.text}\"  score={r.score:.6f}")


if __name__ == "__main__":
    _main()