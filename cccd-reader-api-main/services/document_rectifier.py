"""Module nắn chỉnh hình học và biến đổi phối cảnh (Perspective Transform) cho giấy tờ tùy thân (CCCD & GPLX).

Giai đoạn 3: Document Rectifier bằng OpenCV.
- Đầu vào: Ảnh gốc (np.ndarray BGR), Bounding Box từ YOLOv8n (x1, y1, x2, y2), loại giấy tờ ('cccd' | 'gplx').
- Các bước:
  1. Mở rộng bounding box với safety margin (mặc định 8%).
  2. Tiền xử lý đa tầng: Grayscale, Gaussian Blur / Bilateral Filter, Canny, Dilation & Closing.
  3. Tìm 4 góc vật lý của thẻ: Contour Approx -> Convex Hull Approx -> MinAreaRect Fallback.
  4. Sắp xếp 4 góc chuẩn hóa: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
  5. Biến đổi phối cảnh (cv2.getPerspectiveTransform + cv2.warpPerspective) về kích thước chuẩn (856 x 540).
  6. Fallback an toàn nếu không tìm được 4 góc hoặc input lỗi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Kích thước chuẩn hóa đầu ra theo chuẩn thẻ ID-1 (tỷ lệ ~1.586)
DEFAULT_TARGET_WIDTH = 856
DEFAULT_TARGET_HEIGHT = 540
DEFAULT_MARGIN = 0.08  # Margin 8% an toàn để không cắt mất viền thẻ


@dataclass
class RectifyResult:
    """Kết quả nắn chỉnh phối cảnh của một tài liệu."""

    success: bool
    rectified_image: np.ndarray
    corners: list[list[int]] | None  # 4 góc trên hệ tọa độ ảnh gốc: [[x, y], ...]
    corners_crop: list[list[int]] | None  # 4 góc trên hệ tọa độ crop có margin
    method: str  # 'contour_approx' | 'convex_hull' | 'min_area_rect' | 'fallback_crop'
    confidence: float  # Điểm số heuristic đánh giá chất lượng góc tìm được (0.0 -> 1.0)
    fallback_used: bool
    error: str | None
    aspect_ratio: float  # Tỷ lệ chiều dài / chiều rộng thực tế của tứ giác
    rotation_deg: float  # Góc nghiêng ước tính (độ)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "corners": self.corners,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "fallback_used": self.fallback_used,
            "error": self.error,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "rotation_deg": round(self.rotation_deg, 2),
            "image_shape": list(self.rectified_image.shape),
        }


class DocumentRectifier:
    """Bộ nắn chỉnh phối cảnh giấy tờ sử dụng OpenCV."""

    def __init__(
        self,
        target_size: tuple[int, int] = (DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT),
        margin: float = DEFAULT_MARGIN,
        min_area_ratio: float = 0.20,
        max_area_ratio: float = 0.98,
    ) -> None:
        """Khởi tạo DocumentRectifier.

        Args:
            target_size: (target_width, target_height) mặc định (856, 540).
            margin: Tỷ lệ mở rộng bounding box (0.05 -> 0.15).
            min_area_ratio: Tỷ lệ diện tích tối thiểu của thẻ so với crop có margin.
            max_area_ratio: Tỷ lệ diện tích tối đa để tránh bắt nhầm khung viền ảnh.
        """
        self.target_width, self.target_height = target_size
        self.margin = float(margin)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)

    def rectify(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int] | list[int],
        card_type: str = "cccd",
    ) -> RectifyResult:
        """Thực thi tìm 4 góc và nắn phẳng giấy tờ về kích thước chuẩn.

        Args:
            image: Ảnh gốc BGR dạng NumPy array.
            bbox: Bounding box YOLO [x1, y1, x2, y2].
            card_type: 'cccd' hoặc 'gplx'.

        Returns:
            RectifyResult chứa ảnh đã nắn phẳng hoặc ảnh fallback.
        """
        # 1. Kiểm tra tính hợp lệ của đầu vào
        validation_error = self._validate_input(image, bbox)
        if validation_error is not None:
            logger.warning(f"Input validation failed: {validation_error}")
            return self._create_fallback_result(image, bbox, error=validation_error)

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 2. Mở rộng bounding box với safety margin
        crop_img, (x1m, y1m, x2m, y2m) = self._crop_with_margin(image, (x1, y1, x2, y2))
        if crop_img is None or crop_img.size == 0:
            return self._create_fallback_result(image, bbox, error="Failed to crop image with margin")

        # 3. Phát hiện 4 góc vật lý của thẻ trên vùng crop
        method, corners_crop, confidence = self._detect_corners(crop_img)

        # 4. Nếu không tìm được 4 góc hợp lệ -> Fallback an toàn
        if corners_crop is None:
            logger.info("Không tìm thấy 4 góc đáng tin cậy. Kích hoạt fallback an toàn về YOLO crop.")
            return self._create_fallback_result(
                image,
                bbox,
                crop_img=crop_img,
                error="No valid 4-corner quadrilateral detected",
            )

        # 5. Sắp xếp 4 góc chuẩn hóa: [TL, TR, BR, BL]
        ordered_crop = self._order_quad_corners(corners_crop)

        # 6. Quy đổi tọa độ góc về hệ tọa độ ảnh gốc
        ordered_orig = ordered_crop + np.array([x1m, y1m], dtype=np.float32)

        # 7. Tính góc xoay và tỷ lệ cạnh thực tế
        aspect_ratio, rotation_deg = self._compute_geometry_metrics(ordered_crop)

        # 8. Thực hiện phép biến đổi phối cảnh (Perspective Warp)
        dst_pts = np.array(
            [
                [0, 0],
                [self.target_width - 1, 0],
                [self.target_width - 1, self.target_height - 1],
                [0, self.target_height - 1],
            ],
            dtype=np.float32,
        )

        try:
            matrix = cv2.getPerspectiveTransform(ordered_orig, dst_pts)
            warped = cv2.warpPerspective(
                image,
                matrix,
                (self.target_width, self.target_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception as e:
            logger.warning(f"warpPerspective thất bại: {e}. Kích hoạt fallback.")
            return self._create_fallback_result(
                image, bbox, crop_img=crop_img, error=f"warpPerspective failed: {e}"
            )

        corners_orig_list = [[int(round(pt[0])), int(round(pt[1]))] for pt in ordered_orig]
        corners_crop_list = [[int(round(pt[0])), int(round(pt[1]))] for pt in ordered_crop]

        return RectifyResult(
            success=True,
            rectified_image=warped,
            corners=corners_orig_list,
            corners_crop=corners_crop_list,
            method=method,
            confidence=confidence,
            fallback_used=False,
            error=None,
            aspect_ratio=aspect_ratio,
            rotation_deg=rotation_deg,
        )

    def _validate_input(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int] | list[int],
    ) -> str | None:
        """Kiểm tra tính hợp lệ của ảnh và bounding box."""
        if not isinstance(image, np.ndarray):
            return f"Image must be np.ndarray, got {type(image)}"
        if image.ndim != 3 or image.shape[2] != 3:
            return f"Image must be 3-channel BGR, got shape {image.shape}"
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return f"Image too small or empty: {image.shape}"

        if not hasattr(bbox, "__len__") or len(bbox) != 4:
            return f"BBox must have 4 elements (x1, y1, x2, y2), got {bbox}"

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (ValueError, TypeError):
            return f"BBox coordinates must be integers, got {bbox}"

        h, w = image.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return f"Invalid bbox dimensions (x2 <= x1 or y2 <= y1): {bbox}"
        if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
            return f"BBox completely outside image boundaries ({w}x{h}): {bbox}"

        return None

    def _crop_with_margin(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
        """Mở rộng bounding box với safety margin và trích xuất crop."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        margin_x = int(round(bw * self.margin))
        margin_y = int(round(bh * self.margin))

        x1m = max(0, x1 - margin_x)
        y1m = max(0, y1 - margin_y)
        x2m = min(w, x2 + margin_x)
        y2m = min(h, y2 + margin_y)

        if x2m <= x1m or y2m <= y1m:
            return None, (0, 0, 0, 0)

        crop = image[y1m:y2m, x1m:x2m].copy()
        return crop, (x1m, y1m, x2m, y2m)

    def _detect_corners(
        self,
        crop: np.ndarray,
    ) -> tuple[str, np.ndarray | None, float]:
        """Phát hiện 4 góc vật lý của thẻ bằng đa tầng tiền xử lý và xấp xỉ đa giác.

        Returns:
            (method, corners_crop_float32, confidence)
        """
        h_c, w_c = crop.shape[:2]
        crop_area = float(h_c * w_c)
        if crop_area <= 0:
            return "fallback_none", None, 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Cấu hình đa tầng bộ lọc (Blur size, Canny low, Canny high, Morph kernel, Dilate iter)
        configs = [
            (5, 30, 100, 5, 2),  # Cấu hình chuẩn thẻ ID viền rõ
            (5, 50, 150, 5, 1),  # Thẻ độ tương phản cao
            (7, 20, 80, 7, 2),   # Thẻ ảnh mờ, đường biên đứt đoạn
            (3, 40, 120, 3, 2),  # Thẻ chi tiết mảnh
        ]

        best_pts: np.ndarray | None = None
        best_method = "fallback_none"
        best_score = -1.0

        for blur_k, canny_l, canny_h, morph_k, dilate_iter in configs:
            blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
            edges = cv2.Canny(blurred, canny_l, canny_h)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
            dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Sắp xếp các contour theo diện tích giảm dần
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area < self.min_area_ratio * crop_area or area > self.max_area_ratio * crop_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri <= 0:
                    continue

                # 1. Thử nghiệm xấp xỉ đa giác trực tiếp trên contour với các mức epsilon
                for eps in np.linspace(0.015, 0.08, 14):
                    app = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(app) == 4 and cv2.isContourConvex(app):
                        pts = app.reshape(4, 2).astype(np.float32)
                        score = self._evaluate_quad_candidate(pts, crop_area, area)
                        if score > best_score:
                            best_score = score
                            best_pts = pts
                            best_method = "contour_approx"

                # 2. Thử nghiệm xấp xỉ đa giác trên Convex Hull
                hull = cv2.convexHull(cnt)
                peri_h = cv2.arcLength(hull, True)
                area_h = cv2.contourArea(hull)

                for eps in np.linspace(0.015, 0.08, 14):
                    app_h = cv2.approxPolyDP(hull, eps * peri_h, True)
                    if len(app_h) == 4 and cv2.isContourConvex(app_h):
                        pts = app_h.reshape(4, 2).astype(np.float32)
                        score = self._evaluate_quad_candidate(pts, crop_area, area_h) * 0.95
                        if score > best_score:
                            best_score = score
                            best_pts = pts
                            best_method = "convex_hull"

            if best_pts is not None and best_score > 0.55:
                # Đã tìm được ứng viên có điểm số tốt, dừng quét các config sau
                break

        if best_pts is not None and best_score > 0.35:
            return best_method, best_pts, min(1.0, float(best_score))

        # 3. Fallback cấp 2: Dùng minAreaRect trên contour lớn nhất nếu diện tích đủ lớn
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            c0 = max(cnts, key=cv2.contourArea)
            area0 = cv2.contourArea(c0)
            if area0 > 0.30 * crop_area:
                rect = cv2.minAreaRect(c0)
                box = cv2.boxPoints(rect).astype(np.float32)
                score = (area0 / crop_area) * 0.70
                return "min_area_rect", box, min(1.0, float(score))

        return "fallback_none", None, 0.0

    def _evaluate_quad_candidate(
        self,
        pts: np.ndarray,
        crop_area: float,
        cnt_area: float,
    ) -> float:
        """Đánh giá chất lượng của tứ giác ứng viên (heuristic score)."""
        # Kiểm tra tính lồi
        if not cv2.isContourConvex(pts.astype(np.int32)):
            return 0.0

        # Kiểm tra diện tích tứ giác
        quad_area = cv2.contourArea(pts)
        if quad_area < 0.20 * crop_area or quad_area > 0.98 * crop_area:
            return 0.0

        # Kiểm tra 4 góc trong (phải nằm trong khoảng 50° đến 130°)
        angles = []
        for i in range(4):
            p_prev = pts[(i - 1) % 4]
            p_curr = pts[i]
            p_next = pts[(i + 1) % 4]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
            angles.append(angle)

            if angle < 45.0 or angle > 135.0:
                return 0.0  # Tứ giác quá xiên xẹo hoặc dị thường

        # Tính tỷ lệ cạnh
        e0 = np.linalg.norm(pts[1] - pts[0])
        e1 = np.linalg.norm(pts[2] - pts[1])
        if e0 == 0 or e1 == 0:
            return 0.0

        aspect = max(e0 / e1, e1 / e0)
        # Chuẩn thẻ là ~1.586. Cho phép khoảng 1.1 đến 2.3
        if aspect < 1.1 or aspect > 2.3:
            return 0.0

        # Điểm số kết hợp giữa diện tích, độ vuông góc và tỷ lệ cạnh
        area_score = quad_area / crop_area
        angle_dev = np.mean([abs(a - 90.0) for a in angles])
        angle_score = max(0.0, 1.0 - (angle_dev / 45.0))
        ratio_score = max(0.0, 1.0 - abs(aspect - 1.586) / 1.0)

        total_score = (area_score * 0.45) + (angle_score * 0.35) + (ratio_score * 0.20)
        return float(total_score)

    def _order_quad_corners(self, pts: np.ndarray) -> np.ndarray:
        """Sắp xếp 4 điểm thành [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

        Đảm bảo cạnh (TL -> TR) tương ứng với chiều ngang chuẩn (chiều dài thẻ).
        """
        pts = np.array(pts, dtype=np.float32)

        # 1. Tính trọng tâm đa giác
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])

        # 2. Sắp xếp các điểm theo thứ tự chiều kim đồng hồ quanh trọng tâm
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(angles)
        pts_cw = pts[order]

        # 3. Tính độ dài 4 cạnh liên tiếp: (0->1), (1->2), (2->3), (3->0)
        e_lens = [np.linalg.norm(pts_cw[(i + 1) % 4] - pts_cw[i]) for i in range(4)]

        # Tìm 2 cạnh dài (cạnh đối diện: cặp (0, 2) hoặc cặp (1, 3))
        pair1_len = (e_lens[0] + e_lens[2]) / 2.0
        pair2_len = (e_lens[1] + e_lens[3]) / 2.0

        if pair1_len >= pair2_len:
            # Cặp (0->1) và (2->3) là cạnh dài
            mid_0 = (pts_cw[0] + pts_cw[1]) / 2.0
            mid_2 = (pts_cw[2] + pts_cw[3]) / 2.0
            start_idx = 0 if mid_0[1] <= mid_2[1] else 2
        else:
            # Cặp (1->2) và (3->0) là cạnh dài
            mid_1 = (pts_cw[1] + pts_cw[2]) / 2.0
            mid_3 = (pts_cw[3] + pts_cw[0]) / 2.0
            start_idx = 1 if mid_1[1] <= mid_3[1] else 3

        # Đặt 4 đỉnh theo chiều kim đồng hồ bắt đầu từ start_idx
        tl = pts_cw[start_idx]
        tr = pts_cw[(start_idx + 1) % 4]
        br = pts_cw[(start_idx + 2) % 4]
        bl = pts_cw[(start_idx + 3) % 4]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _compute_geometry_metrics(self, ordered_pts: np.ndarray) -> tuple[float, float]:
        """Tính tỷ lệ aspect_ratio và góc nghiêng ước tính (rotation_deg)."""
        tl, tr, br, bl = ordered_pts

        top_w = np.linalg.norm(tr - tl)
        bot_w = np.linalg.norm(br - bl)
        w = (top_w + bot_w) / 2.0

        left_h = np.linalg.norm(bl - tl)
        right_h = np.linalg.norm(br - tr)
        h = (left_h + right_h) / 2.0

        aspect = float(w / h) if h > 0 else 0.0

        # Góc của cạnh trên (TL -> TR) so với trục hoành
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        rotation = float(np.degrees(np.arctan2(dy, dx)))

        return aspect, rotation

    def _create_fallback_result(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int] | list[int],
        crop_img: np.ndarray | None = None,
        error: str | None = None,
    ) -> RectifyResult:
        """Tạo đối tượng RectifyResult an toàn khi thất bại (fallback về YOLO crop)."""
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            resized_fallback = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            return RectifyResult(
                success=False,
                rectified_image=resized_fallback,
                corners=None,
                corners_crop=None,
                method="fallback_crop",
                confidence=0.0,
                fallback_used=True,
                error=error or "Invalid image input",
                aspect_ratio=float(self.target_width / self.target_height),
                rotation_deg=0.0,
            )

        h_orig, w_orig = image.shape[:2]

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            if x2 > x1 and y2 > y1:
                fallback_img = image[y1:y2, x1:x2].copy()
            else:
                fallback_img = image.copy()
        except Exception:
            fallback_img = image.copy()

        # Resize nhẹ fallback về target size nếu cần đảm bảo shape nhất quán
        if fallback_img.size > 0:
            resized_fallback = cv2.resize(
                fallback_img, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized_fallback = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        return RectifyResult(
            success=False,
            rectified_image=resized_fallback,
            corners=None,
            corners_crop=None,
            method="fallback_crop",
            confidence=0.0,
            fallback_used=True,
            error=error,
            aspect_ratio=float(self.target_width / self.target_height),
            rotation_deg=0.0,
        )

    def draw_corners(
        self,
        image: np.ndarray,
        corners: list[list[int]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Tiện ích vẽ 4 góc và đường viền tứ giác lên ảnh để trực quan hóa debug."""
        annotated = image.copy()
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)

        labels = ["TL", "TR", "BR", "BL"]
        colors_pts = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]

        for i, (pt, label, pt_col) in enumerate(zip(corners, labels, colors_pts)):
            cv2.circle(annotated, tuple(pt), 6, pt_col, -1)
            cv2.putText(
                annotated,
                label,
                (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                pt_col,
                2,
                cv2.LINE_AA,
            )

        return annotated
