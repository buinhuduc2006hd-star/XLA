"""Kịch bản huấn luyện YOLOv8-Pose (Ultralytics) cho bài toán định vị 4 góc thẻ CCCD & GPLX.

Đặc điểm an toàn:
1. Không tự động chạy huấn luyện khi được import (chỉ chạy qua CLI với if __name__ == '__main__':).
2. Kiểm tra tính toàn vẹn của dataset trước khi train (dừng an toàn nếu dataset rỗng).
3. Hỗ trợ linh hoạt thiết bị CUDA GPU và CPU Windows 11.
4. Lưu checkpoint riêng biệt tại runs/pose/ hoặc models/document_pose/, TUYỆT ĐỐI không ghi đè model Detection cũ (models/yolov8n_document.pt).
5. Hỗ trợ siêu tham số tối ưu cho thẻ ID-1: kpt_shape=[4, 3], image_size=640, Early Stopping patience=20.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# Đảm bảo import được các module từ scripts/pose
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from validate_pose_dataset import validate_dataset
except ImportError:
    validate_dataset = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pose")


def _resolve_candidate_path(raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    candidates = [
        Path.cwd() / p,
        Path.cwd() / "cccd-reader-api-main" / p,
        Path(__file__).resolve().parent.parent.parent / p,
    ]
    parts = p.parts
    if len(parts) > 1 and parts[0] == "cccd-reader-api-main":
        candidates.insert(0, Path.cwd() / Path(*parts[1:]))
    for c in candidates:
        if c.exists():
            return c.resolve()
    return p.resolve()


def train_yolov8_pose(
    data_yaml: str | Path,
    base_model: str = "yolov8n-pose.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    project: str = "runs/pose",
    name: str = "document_pose_v1",
    patience: int = 20,
    workers: int = 4,
    dest_model_dir: str | Path | None = None,
    resume: bool = False,
) -> Path | None:
    """Thực thi huấn luyện YOLOv8-Pose từ dataset chuẩn bị.

    Args:
        data_yaml: Đường dẫn file card_pose.yaml.
        base_model: Trọng số khởi tạo pre-trained (mặc định 'yolov8n-pose.pt').
        epochs: Số lượng epoch tối đa (mặc định 100).
        batch_size: Kích thước batch (mặc định 16).
        imgsz: Kích thước ảnh đầu vào (mặc định 640).
        device: 'auto', '0', 'cpu', hoặc 'cuda'.
        project: Thư mục gốc lưu kết quả training.
        name: Tên phiên huấn luyện.
        patience: Số epoch không cải thiện trước khi dừng sớm (Early Stopping).
        workers: Số worker nạp dữ liệu.
        dest_model_dir: Thư mục đích để copy checkpoint best.pt sau khi hoàn tất.

    Returns:
        Đường dẫn tới file checkpoint 'best.pt' nếu huấn luyện thành công.
    """
    yaml_p = _resolve_candidate_path(data_yaml)
    if not yaml_p.exists():
        logger.error(f"File cấu hình dataset không tồn tại: {yaml_p}")
        return None

    # 1. Kiểm tra dataset có đủ dữ liệu không
    ds_dir = yaml_p.parent
    if validate_dataset is not None:
        report = validate_dataset(ds_dir, yaml_p)
        if not report["is_ready_for_training"]:
            logger.error("Dataset chưa sẵn sàng để huấn luyện!")
            logger.error(f"Tổng số ảnh: {report['total_images']}. Lỗi: {len(report['errors'])}")
            logger.error("Vui lòng gắn nhãn và thêm dữ liệu vào images/train và labels/train trước khi chạy.")
            return None

    # 2. Xác định thiết bị tính toán
    import torch
    if device == "auto":
        dev = "0" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    logger.info(f"Thiết bị huấn luyện: {dev} (CUDA Available: {torch.cuda.is_available()})")

    # 3. Khởi tạo mô hình Ultralytics YOLO
    logger.info(f"Khởi tạo mô hình pre-trained: {base_model}")
    from ultralytics import YOLO
    model = YOLO(base_model)

    # 4. Cấu hình siêu tham số
    # Với pose 4 góc thẻ ID-1:
    # - Hạn chế shear / perspective quá đà làm méo tứ giác
    # - Tăng cường mosaic, flip left-right, color jitter nhẹ
    logger.info(f"Bắt đầu huấn luyện: epochs={epochs}, batch={batch_size}, imgsz={imgsz}, patience={patience}, resume={resume}")
    train_results = model.train(
        data=str(yaml_p),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=dev,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        resume=resume,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # Augmentation phù hợp với giấy tờ tùy thân
        fliplr=0.5,
        flipud=0.0,  # Không lật trên-dưới (thẻ có chiều chuẩn)
        mosaic=1.0,
        mixup=0.1,
        degrees=15.0,  # Xoay nhẹ +/- 15 độ
        shear=2.0,
        perspective=0.0005,
    )

    best_checkpoint = Path(project) / name / "weights" / "best.pt"
    if not best_checkpoint.exists():
        logger.warning(f"Không tìm thấy checkpoint 'best.pt' tại {best_checkpoint}")
        return None

    logger.info(f"Huấn luyện hoàn tất thành công! Checkpoint tốt nhất: {best_checkpoint}")

    # Copy tới thư mục đích nếu được chỉ định
    if dest_model_dir:
        dest_p = Path(dest_model_dir).resolve()
        dest_p.mkdir(parents=True, exist_ok=True)
        target_file = dest_p / "best.pt"
        shutil.copy2(best_checkpoint, target_file)
        logger.info(f"Đã sao chép checkpoint sang: {target_file}")
        return target_file

    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Pose for Vietnamese ID card corner detection.")
    parser.add_argument(
        "--data",
        type=str,
        default="cccd-reader-api-main/datasets/document_pose/card_pose.yaml",
        help="Path to card_pose.yaml",
    )
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Base pretrained pose weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", help="Computing device ('auto', 'cpu', '0')")
    parser.add_argument("--project", type=str, default="cccd-reader-api-main/runs/pose", help="Project output folder")
    parser.add_argument("--name", type=str, default="document_pose_v1", help="Run name")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (2 on Windows to avoid freeze)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["test30", "full100"],
        help="Quick configuration profile: 'test30' (30 epochs test) or 'full100' (100 epochs production)",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="cccd-reader-api-main/models/document_pose",
        help="Directory to copy best.pt after training",
    )
    args = parser.parse_args()

    epochs = args.epochs
    if args.profile == "test30":
        epochs = 30
        logger.info("Sử dụng profile 'test30': Huấn luyện 30 epochs thử nghiệm.")
    elif args.profile == "full100":
        epochs = 100
        logger.info("Sử dụng profile 'full100': Huấn luyện 100 epochs chính thức.")

    train_yolov8_pose(
        data_yaml=args.data,
        base_model=args.model,
        epochs=epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        dest_model_dir=args.dest_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
