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

    def rectify_from_corners(
        self,
        image: np.ndarray,
        corners: np.ndarray | Sequence[Sequence[float]],
        card_type: str = "cccd",
        target_size: tuple[int, int] | None = None,
    ) -> RectifyResult:
        """Nắn phẳng giấy tờ trực tiếp từ 4 góc vật lý dự đoán (từ YOLOv8-Pose).

        Không cần chạy lại các thuật toán OpenCV Contour/Edge Detection.
        Đảm bảo không lật gương và chuẩn hóa về kích thước thẻ ID-1 (856 x 540).

        Args:
            image: Ảnh gốc BGR dạng NumPy array.
            corners: Mảng 4 góc theo quy ước Roboflow:
                     Index 0: point_up_left (TL)
                     Index 1: point_up_right (TR)
                     Index 2: point_bottom_right (BR)
                     Index 3: point_bottom_left (BL)
            card_type: 'cccd' hoặc 'gplx'.
            target_size: (target_width, target_height), mặc định (856, 540).

        Returns:
            RectifyResult chứa ảnh đã nắn phẳng chuẩn tỉ lệ.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
            return RectifyResult(
                success=False,
                rectified_image=np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8),
                corners=None,
                corners_crop=None,
                method="pose_keypoints",
                confidence=0.0,
                fallback_used=True,
                error="Invalid or empty input image matrix",
                aspect_ratio=0.0,
                rotation_deg=0.0,
            )

        pts = np.asarray(corners, dtype=np.float32)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            return RectifyResult(
                success=False,
                rectified_image=image.copy(),
                corners=None,
                corners_crop=None,
                method="pose_keypoints",
                confidence=0.0,
                fallback_used=True,
                error=f"Invalid corners shape {pts.shape} or contains non-finite values",
                aspect_ratio=0.0,
                rotation_deg=0.0,
            )

        # Kiểm tra không lật gương / đảo chiều (Vector TL->TR x TL->BL > 0)
        v_top = pts[1] - pts[0]
        v_left = pts[3] - pts[0]
        cross_z = v_top[0] * v_left[1] - v_top[1] * v_left[0]
        if cross_z <= 0:
            logger.warning(f"4 góc bị lật gương (cross_z={cross_z:.1f} <= 0). Tự động đảo chiều [TL, BL, BR, TR].")
            # Nếu bị lật ngược thứ tự, chuẩn hóa lại
            pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)

        target_w, target_h = target_size or (self.target_width, self.target_height)
        dst_pts = np.array(
            [
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1],
            ],
            dtype=np.float32,
        )

        aspect_ratio, rotation_deg = self._compute_geometry_metrics(pts)

        try:
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            warped = cv2.warpPerspective(
                image,
                matrix,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception as e:
            logger.error(f"warpPerspective từ 4 keypoints thất bại: {e}")
            return RectifyResult(
                success=False,
                rectified_image=image.copy(),
                corners=pts.astype(int).tolist(),
                corners_crop=None,
                method="pose_keypoints",
                confidence=0.0,
                fallback_used=True,
                error=f"warpPerspective failed: {e}",
                aspect_ratio=aspect_ratio,
                rotation_deg=rotation_deg,
            )

        corners_list = [[int(round(p[0])), int(round(p[1]))] for p in pts]
        return RectifyResult(
            success=True,
            rectified_image=warped,
            corners=corners_list,
            corners_crop=None,
            method="pose_keypoints",
            confidence=1.0,
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

        # 3. Chiến lược 3: Phát hiện 4 cạnh bằng Hough Lines P và tìm giao điểm (trị góc bo tròn)
        line_pts = self._detect_corners_by_lines(gray, crop_area)
        if line_pts is not None:
            score_lines = self._evaluate_quad_candidate(line_pts, crop_area, crop_area) * 0.90
            if score_lines > best_score:
                best_score = score_lines
                best_pts = line_pts
                best_method = "line_intersection"

        if best_pts is not None and best_score >= 0.38:
            return best_method, best_pts, min(1.0, float(best_score))

        # Yêu cầu 5: Tuyệt đối không lấy 4 góc bounding box (min_area_rect) làm 4 góc thẻ!
        # Nếu OpenCV không tìm được tứ giác 4 góc vật lý thực tế -> Báo thất bại
        return "fallback_none", None, 0.0

    def _detect_corners_by_lines(self, gray: np.ndarray, crop_area: float) -> np.ndarray | None:
        """Tìm 4 góc vật lý dựa trên giao điểm của 4 cạnh chính (Hough Lines).

        Hữu hiệu khi 4 góc thẻ bị bo tròn (radius 3.18mm) khiến contour approximation bị cắt vụn.
        """
        h_c, w_c = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        min_len = int(min(h_c, w_c) * 0.20)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=min_len, maxLineGap=20)
        if lines is None or len(lines) < 4:
            return None

        top_pts, bot_pts, left_pts, right_pts = [], [], [], []
        cy, cx = h_c / 2.0, w_c / 2.0

        for line in lines:
            pts_line = line.reshape(-1)
            if len(pts_line) < 4:
                continue
            x1, y1, x2, y2 = int(pts_line[0]), int(pts_line[1]), int(pts_line[2]), int(pts_line[3])
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length < min_len:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            if angle <= 30.0 or angle >= 150.0:
                mid_y = (y1 + y2) / 2.0
                if mid_y < cy:
                    top_pts.extend([[x1, y1], [x2, y2]])
                else:
                    bot_pts.extend([[x1, y1], [x2, y2]])
            elif 60.0 <= angle <= 120.0:
                mid_x = (x1 + x2) / 2.0
                if mid_x < cx:
                    left_pts.extend([[x1, y1], [x2, y2]])
                else:
                    right_pts.extend([[x1, y1], [x2, y2]])

        if len(top_pts) < 2 or len(bot_pts) < 2 or len(left_pts) < 2 or len(right_pts) < 2:
            return None

        def fit_l(pts_list):
            arr = np.array(pts_list, dtype=np.float32)
            l = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01)
            return (float(l[0]), float(l[1]), float(l[2]), float(l[3]))

        def intersect(l1, l2):
            vx1, vy1, x1, y1 = l1
            vx2, vy2, x2, y2 = l2
            A = np.array([[-vy1, vx1], [-vy2, vx2]], dtype=np.float32)
            B = np.array([vx1 * y1 - vy1 * x1, vx2 * y2 - vy2 * x2], dtype=np.float32)
            det = float(np.linalg.det(A))
            if abs(det) < 1e-4:
                return None
            return np.linalg.solve(A, B)

        try:
            l_top = fit_l(top_pts)
            l_bot = fit_l(bot_pts)
            l_left = fit_l(left_pts)
            l_right = fit_l(right_pts)

            tl = intersect(l_top, l_left)
            tr = intersect(l_top, l_right)
            br = intersect(l_bot, l_right)
            bl = intersect(l_bot, l_left)

            if tl is None or tr is None or br is None or bl is None:
                return None

            quad = np.array([tl, tr, br, bl], dtype=np.float32)
            margin_tol = 0.15
            if (
                np.any(quad[:, 0] < -margin_tol * w_c)
                or np.any(quad[:, 0] > (1 + margin_tol) * w_c)
                or np.any(quad[:, 1] < -margin_tol * h_c)
                or np.any(quad[:, 1] > (1 + margin_tol) * h_c)
            ):
                return None

            quad[:, 0] = np.clip(quad[:, 0], 0, w_c - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, h_c - 1)
            return quad
        except Exception:
            return None

    def _evaluate_quad_candidate(
        self,
        pts: np.ndarray,
        crop_area: float,
        cnt_area: float,
    ) -> float:
        """Đánh giá chất lượng của tứ giác ứng viên theo các ràng buộc vật lý thực tế của thẻ ID-1."""
        # 1. Kiểm tra tính lồi
        if not cv2.isContourConvex(pts.astype(np.int32)):
            return 0.0

        # 2. Kiểm tra diện tích tứ giác so với crop (thẻ phải chiếm tỷ lệ hợp lý, không bắt chi tiết nhỏ)
        quad_area = cv2.contourArea(pts)
        if quad_area < 0.28 * crop_area or quad_area > 0.95 * crop_area:
            return 0.0

        # 3. Kiểm tra 4 góc trong (phải nằm trong khoảng 50° đến 130°)
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

            if angle < 50.0 or angle > 130.0:
                return 0.0  # Tứ giác quá xiên xẹo hoặc dị thường

        # 4. Kiểm tra tỷ lệ cạnh vật lý (chuẩn ID-1 là 85.6/53.98 ~ 1.586)
        # Giới hạn vật lý khi chiếu phối cảnh thông thường: 1.25 -> 2.05.
        # Loại bỏ các tứ giác gần vuông (aspect ~ 1.0) hoặc dẹp bất thường (> 2.1)
        e0 = np.linalg.norm(pts[1] - pts[0])
        e1 = np.linalg.norm(pts[2] - pts[1])
        if e0 == 0 or e1 == 0:
            return 0.0

        aspect = max(e0 / e1, e1 / e0)
        if aspect < 1.25 or aspect > 2.05:
            return 0.0

        # 5. Điểm số kết hợp giữa diện tích, độ vuông góc và tỷ lệ cạnh
        area_score = quad_area / crop_area
        angle_dev = np.mean([abs(a - 90.0) for a in angles])
        angle_score = max(0.0, 1.0 - (angle_dev / 40.0))
        ratio_score = max(0.0, 1.0 - abs(aspect - 1.586) / 0.8)

        total_score = (area_score * 0.40) + (angle_score * 0.35) + (ratio_score * 0.25)
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
        title: str | None = None,
    ) -> np.ndarray:
        """Tiện ích vẽ 4 góc và đường viền tứ giác lên ảnh để trực quan hóa debug.

        Đánh dấu rõ ràng 4 góc:
        - TL (Top-Left): Đỏ (Red)
        - TR (Top-Right): Vàng (Yellow)
        - BR (Bottom-Right): Xanh dương (Blue)
        - BL (Bottom-Left): Tím/Hồng (Magenta)
        """
        annotated = image.copy()
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)

        labels = ["TL", "TR", "BR", "BL"]
        # Thứ tự màu BGR: TL=Red (0,0,255), TR=Yellow (0,255,255), BR=Blue (255,120,0), BL=Magenta (255,0,255)
        colors_pts = [(0, 0, 255), (0, 255, 255), (255, 120, 0), (255, 0, 255)]

        for i, (pt, label, pt_col) in enumerate(zip(corners, labels, colors_pts)):
            # Chấm tròn tâm và vòng tròn bảo vệ bên ngoài
            cv2.circle(annotated, tuple(pt), 7, pt_col, -1)
            cv2.circle(annotated, tuple(pt), 10, (255, 255, 255), 2)
            cv2.putText(
                annotated,
                f"{label} ({pt[0]},{pt[1]})",
                (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                pt_col,
                2,
                cv2.LINE_AA,
            )

        if title:
            # Banner thông tin phía trên
            cv2.rectangle(annotated, (10, 10), (min(image.shape[1] - 10, 680), 45), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                title,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated
