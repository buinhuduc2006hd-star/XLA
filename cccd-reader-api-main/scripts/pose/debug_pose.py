"""Kịch bản trực quan hóa kết quả YOLOv8-Pose (Debug Tool).

Chức năng:
1. Nạp ảnh đầu vào.
2. Chạy suy luận qua YOLOv8-Pose model.
3. Vẽ Bounding Box, Class Name và Detection Confidence.
4. Vẽ 4 góc vật lý [TL, TR, BR, BL] với mã màu và nhãn trực quan.
5. Vẽ đa giác tứ giác kết nối 4 góc.
6. Thực hiện Perspective Transform nắn phẳng về 856x540.
7. Lưu ảnh debug và ảnh nắn phẳng vào thư mục debug (mặc định debug_output/pose/).
8. Tuyệt đối không log thông tin cá nhân (PII) ra console.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Thiết lập đường dẫn import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_pose_detector import DocumentPoseDetector, validate_card_keypoints
from services.document_rectifier import DocumentRectifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_pose")

# Màu sắc BGR chuẩn cho 4 góc
CORNER_COLORS = {
    0: (0, 0, 255),    # TL: Đỏ
    1: (0, 255, 255),  # TR: Vàng
    2: (255, 0, 0),    # BR: Xanh dương
    3: (255, 0, 255),  # BL: Tím
}
ROBOFLOW_KEYPOINT_NAMES = [
    "point_up_left",
    "point_up_right",
    "point_bottom_right",
    "point_bottom_left",
]
KEYPOINT_LABELS = [
    "point_up_left (TL)",
    "point_up_right (TR)",
    "point_bottom_right (BR)",
    "point_bottom_left (BL)",
]


def draw_pose_debug_visualization(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    corners: np.ndarray | None,
    doc_type: str,
    conf: float,
    kpt_confs: list[float] | None = None,
) -> np.ndarray:
    """Vẽ bounding box, 4 góc và polygon kết nối lên bản sao của ảnh gốc."""
    vis = image.copy()
    h_img, w_img = vis.shape[:2]

    # 1. Vẽ Bounding Box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{doc_type.upper()} ({conf:.2f})"
        cv2.putText(vis, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. Vẽ 4 góc và đường bao tứ giác
    if corners is not None and len(corners) == 4:
        pts = corners.astype(np.int32)

        # Vẽ đường viền đa giác tứ giác kết nối 4 góc
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 220, 255), thickness=2)

        for i in range(4):
            cx, cy = int(round(corners[i][0])), int(round(corners[i][1]))
            color = CORNER_COLORS.get(i, (255, 255, 255))
            kpt_lbl = KEYPOINT_LABELS[i]
            kpt_c = f"{kpt_confs[i]:.2f}" if (kpt_confs and i < len(kpt_confs)) else ""

            # Vòng tròn kép nổi bật
            cv2.circle(vis, (cx, cy), 8, color, -1)
            cv2.circle(vis, (cx, cy), 12, (255, 255, 255), 2)

            # Nhãn góc
            text = f"{kpt_lbl} {kpt_c}".strip()
            # Đặt vị trí text offset nhẹ để không che tâm góc
            tx = cx + 14 if i in (1, 2) else cx - 60
            ty = cy - 10 if i in (0, 1) else cy + 20
            tx = max(5, min(w_img - 70, tx))
            ty = max(20, min(h_img - 10, ty))

            cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


def run_debug_pose(
    image_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path = "debug_output/pose",
    expected_type: str = "auto",
) -> None:
    """Chạy debug YOLOv8-Pose trên 1 ảnh và lưu trực quan hóa."""
    img_p = Path(image_path).resolve()
    if not img_p.exists():
        logger.error(f"Ảnh không tồn tại: {img_p}")
        return

    m_p = Path(model_path).resolve()
    if not m_p.exists():
        logger.error(f"Checkpoint mô hình không tồn tại: {m_p}")
        return

    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        logger.error(f"Không thể đọc file ảnh: {img_p}")
        return

    out_d = Path(output_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    detector = DocumentPoseDetector(model_path=m_p)
    pose_res = detector.detect(img_bgr, expected_type=expected_type)

    logger.info(f"Kết quả Pose: success={pose_res.success}, type={pose_res.document_type}, conf={pose_res.confidence}")
    if not pose_res.success:
        logger.warning(f"Thất bại: {pose_res.failure_reason}")

    # Vẽ visualization
    vis_img = draw_pose_debug_visualization(
        image=img_bgr,
        bbox=pose_res.bbox,
        corners=pose_res.corners,
        doc_type=pose_res.document_type or "unknown",
        conf=pose_res.confidence or 0.0,
        kpt_confs=pose_res.keypoint_confidences,
    )

    stem = img_p.stem
    vis_out_path = out_d / f"{stem}_pose_debug.jpg"
    cv2.imwrite(str(vis_out_path), vis_img)
    logger.info(f"Đã lưu ảnh debug: {vis_out_path}")

    # Nếu thành công nắn phối cảnh
    if pose_res.success and pose_res.corners is not None:
        rectifier = DocumentRectifier()
        rect_res = rectifier.rectify_from_corners(img_bgr, pose_res.corners, card_type=pose_res.document_type or "cccd")
        if rect_res.success:
            rect_out_path = out_d / f"{stem}_rectified.jpg"
            cv2.imwrite(str(rect_out_path), rect_res.rectified_image)
            logger.info(f"Đã lưu ảnh nắn phẳng: {rect_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug and visualize YOLOv8-Pose detection on Vietnamese ID cards.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="cccd-reader-api-main/models/document_pose/best.pt",
        help="Path to YOLOv8-Pose checkpoint",
    )
    parser.add_argument("--output-dir", type=str, default="debug_output/pose", help="Output directory for debug images")
    parser.add_argument("--type", type=str, default="auto", help="Expected document type ('auto', 'cccd', 'gplx')")
    args = parser.parse_args()

    run_debug_pose(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        expected_type=args.type,
    )


if __name__ == "__main__":
    main()
