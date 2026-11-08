"""Script đánh giá hiệu năng mô hình YOLOv8n trên tập kiểm thử CCCD & GPLX.

Sử dụng:
    python -m scripts.train.evaluate --weights path/to/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_yolov8n_document(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    batch_size: int = 8,
    device: str | None = None,
    save_json: str | Path | None = None,
) -> dict:
    """Đánh giá model trên split dữ liệu (val hoặc test).

    Args:
        weights: Đường dẫn trọng số model .pt.
        data_yaml: Đường dẫn file cấu hình data.yaml.
        split: 'val' hoặc 'test'.
        imgsz: Kích thước ảnh kiểm thử.
        batch_size: Batch size.
        device: 'cpu' hoặc 'cuda'.
        save_json: Đường dẫn lưu kết quả JSON nếu muốn.

    Returns:
        Dict tổng hợp các chỉ số mAP50, mAP50-95, precision, recall, speed.
    """
    weights_path = Path(weights).resolve()
    yaml_path = Path(data_yaml).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {yaml_path}")

    logger.info(f"Đang đánh giá model: {weights_path.name} trên split '{split}'...")
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        verbose=True,
    )

    box_metrics = metrics.box
    class_map = {0: "cccd", 1: "gplx"}

    report = {
        "weights": str(weights_path),
        "split": split,
        "overall": {
            "mAP50": round(float(box_metrics.map50), 4),
            "mAP50_95": round(float(box_metrics.map), 4),
            "precision": round(float(box_metrics.mp), 4),
            "recall": round(float(box_metrics.mr), 4),
        },
        "per_class": {},
        "speed_ms": {
            "preprocess": round(float(metrics.speed.get("preprocess", 0.0)), 2),
            "inference": round(float(metrics.speed.get("inference", 0.0)), 2),
            "postprocess": round(float(metrics.speed.get("postprocess", 0.0)), 2),
        },
    }

    # Trích xuất chỉ số từng lớp
    if hasattr(box_metrics, "maps") and len(box_metrics.maps) > 0:
        for c_id, map_val in enumerate(box_metrics.maps):
            c_name = class_map.get(c_id, f"class_{c_id}")
            report["per_class"][c_name] = {
                "class_id": c_id,
                "mAP50_95": round(float(map_val), 4),
            }

    print("\n==========================================")
    print("KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG YOLOV8N")
    print(f"- Trọng số: {weights_path.name}")
    print(f"- Tập dữ liệu: {split}")
    print(f"- mAP@0.50:       {report['overall']['mAP50']:.4f}")
    print(f"- mAP@0.50:0.95:  {report['overall']['mAP50_95']:.4f}")
    print(f"- Precision:      {report['overall']['precision']:.4f}")
    print(f"- Recall:         {report['overall']['recall']:.4f}")
    print(f"- Tốc độ suy luận: {report['speed_ms']['inference']} ms/ảnh")
    print("==========================================")

    if save_json:
        out_p = Path(save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu kết quả đánh giá vào: {out_p}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình YOLOv8n Document Detection")
    parser.add_argument("--weights", required=True, help="Đường dẫn file trọng số model (.pt)")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent.parent.parent / "configs" / "data.yaml"),
        help="Đường dẫn file data.yaml",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Tập dữ liệu đánh giá")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Device ('cpu' hoặc 'cuda')")
    parser.add_argument("--save-json", default=None, help="Đường dẫn lưu file JSON")

    args = parser.parse_args()

    evaluate_yolov8n_document(
        weights=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        save_json=args.save_json,
    )

