"""Script chạy thử nghiệm nhận diện tài liệu (CCCD & GPLX) trên ảnh đơn hoặc thư mục ảnh.

Sử dụng:
    # 1. Chạy trên một ảnh và lưu kết quả:
    python -m scripts.predict_sample test_images/sample.jpg --weights models/yolov8n_document.pt --save-dir runs/predict

    # 2. Chạy trên toàn bộ thư mục:
    python -m scripts.predict_sample path/to/images_dir --save-dir runs/predict --conf 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Thêm thư mục cccd-reader-api-main vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from services.document_detector import DocumentDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_predict(
    source: str | Path,
    weights: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    save_dir: str | Path | None = "runs/detect/predict",
    show: bool = False,
) -> list[dict]:
    """Thực thi dự đoán tài liệu trên ảnh hoặc thư mục ảnh.

    Args:
        source: Đường dẫn tới 1 file ảnh hoặc 1 thư mục chứa ảnh.
        weights: Đường dẫn file weights .pt (nếu None sẽ tự tìm).
        conf: Ngưỡng confidence.
        iou: Ngưỡng IoU.
        device: 'cpu' hoặc 'cuda'.
        save_dir: Thư mục lưu ảnh đã vẽ bounding box.
        show: Hiển thị cửa sổ OpenCV nếu có màn hình.

    Returns:
        Danh sách kết quả phát hiện cho từng ảnh.
    """
    src_path = Path(source).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source không tồn tại: {src_path}")

    # Thu thập danh sách ảnh
    if src_path.is_file():
        image_files = [src_path]
    else:
        image_files = [f for f in src_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not image_files:
        logger.warning(f"Không tìm thấy file ảnh hợp lệ trong: {src_path}")
        return []

    logger.info(f"Tìm thấy {len(image_files)} ảnh để dự đoán.")

    # Khởi tạo detector
    detector = DocumentDetector(
        model_path=weights,
        conf_threshold=conf,
        iou_threshold=iou,
        device=device,
    )

    out_dir = Path(save_dir).resolve() if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for img_path in image_files:
        logger.info(f"\n--- Đang xử lý: {img_path.name} ---")
        try:
            result = detector.detect(img_path)
            res_dict = {
                "file": img_path.name,
                "has_detection": result.has_detection,
                "count": len(result.detections),
                "detections": [d.to_dict() for d in result.detections],
            }
            all_results.append(res_dict)

            # In kết quả ra console
            if result.has_detection:
                for idx, det in enumerate(result.detections, 1):
                    logger.info(
                        f"  [{idx}] Lớp: {det.class_name.upper()} | "
                        f"Conf: {det.confidence:.4f} | BBox (xyxy): {det.bbox_xyxy}"
                    )
            else:
                logger.info("  [!] Không phát hiện giấy tờ nào vượt ngưỡng confidence.")

            # Vẽ và lưu ảnh kết quả
            if out_dir:
                save_file = out_dir / f"pred_{img_path.name}"
                annotated = detector.annotate_image(img_path, result.detections, save_path=save_file)
                logger.info(f"  -> Đã lưu ảnh kết quả tại: {save_file}")

                if show:
                    cv2.imshow(f"Detection: {img_path.name}", annotated)
                    cv2.waitKey(0)

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán ảnh {img_path.name}: {e}")

    if show:
        cv2.destroyAllWindows()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán và trực quan hóa bounding box CCCD & GPLX")
    parser.add_argument("source", help="Đường dẫn file ảnh hoặc thư mục ảnh")
    parser.add_argument("--weights", default=None, help="Đường dẫn file trọng số YOLO .pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence (0.0-1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng NMS IoU (0.0-1.0)")
    parser.add_argument("--device", default=None, help="Device ('cpu' hoặc 'cuda')")
    parser.add_argument("--save-dir", default="runs/detect/predict", help="Thư mục lưu ảnh đã vẽ bbox")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ xem ảnh")

    args = parser.parse_args()

    run_predict(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=args.save_dir,
        show=args.show,
    )
