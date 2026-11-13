"""Script huấn luyện / fine-tune mô hình YOLOv8n trên dataset nhận diện CCCD & GPLX.

Sử dụng:
    python -m scripts.train.train --epochs 50 --batch 8 --imgsz 640
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_yolov8n_document(
    data_yaml: str | Path,
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str | None = None,
    pretrained_weights: str = "yolov8n.pt",
    project: str = "runs/detect",
    name: str = "train_document",
    patience: int = 15,
    export_dir: str | Path | None = None,
) -> Path | None:
    """Fine-tune YOLOv8n trên dataset giấy tờ tùy thân.

    Args:
        data_yaml: Đường dẫn file data.yaml cấu hình dataset.
        epochs: Số lượng epoch huấn luyện.
        batch_size: Kích thước mini-batch (8 hoặc 16 phù hợp CPU/GPU thông dụng).
        imgsz: Kích thước ảnh chuẩn hóa (mặc định 640).
        device: 'cuda', 'cpu', hoặc None (tự động chọn).
        pretrained_weights: Trọng số khởi tạo nền (mặc định 'yolov8n.pt').
        project: Thư mục gốc lưu kết quả run.
        name: Tên thí nghiệm huấn luyện.
        patience: Số epoch early-stopping nếu loss không giảm.
        export_dir: Thư mục lưu bản sao trọng số tốt nhất (vd: models/).

    Returns:
        Đường dẫn file best.pt đã huấn luyện, hoặc None nếu chưa có dữ liệu.
    """
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {yaml_path}")

    # Tự động chọn thiết bị
    if device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    logger.info("==================================================")
    logger.info("KHỞI TẠO HUẤN LUYỆN YOLOV8N DOCUMENT DETECTOR")
    logger.info(f"- Data config: {yaml_path}")
    logger.info(f"- Trọng số gốc: {pretrained_weights}")
    logger.info(f"- Thiết bị: {target_device} (CUDA available: {torch.cuda.is_available()})")
    logger.info(f"- Cấu hình: epochs={epochs}, batch={batch_size}, imgsz={imgsz}")
    logger.info("==================================================")

    # Khởi tạo mô hình YOLOv8n
    model = YOLO(pretrained_weights)

    try:
        # Huấn luyện mô hình với các siêu tham số phù hợp phát hiện tài liệu
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=target_device,
            project=project,
            name=name,
            patience=patience,
            save=True,
            val=True,
            verbose=True,
            # Các augmentation hỗ trợ phát hiện thẻ xoay/nghiêng
            degrees=15.0,        # Xoay ngẫu nhiên +/- 15 độ
            perspective=0.0005,  # Biến dạng phối cảnh nhẹ
            scale=0.2,           # Thu phóng ngẫu nhiên
            fliplr=0.5,          # Lật gương ngang
            flipud=0.0,          # Không lật dọc thẻ (để giữ hướng chữ)
        )
    except Exception as e:
        logger.error(f"Quá trình huấn luyện gặp lỗi: {e}")
        return None

    # Tìm đường dẫn trọng số tốt nhất sau khi train
    # Ultralytics tự động tạo thư mục run (vd: train_document, train_document2, ...)
    best_weights = None
    if hasattr(model, "trainer") and model.trainer is not None:
        if hasattr(model.trainer, "best") and model.trainer.best:
            best_weights = Path(model.trainer.best)
        elif hasattr(model.trainer, "save_dir") and model.trainer.save_dir:
            cand = Path(model.trainer.save_dir) / "weights" / "best.pt"
            if cand.exists():
                best_weights = cand

    # Fallback: tìm theo thư mục project/name
    if best_weights is None or not best_weights.exists():
        exp_dir = Path(project) / name
        cand = exp_dir / "weights" / "best.pt"
        if cand.exists():
            best_weights = cand
        else:
            # Tìm tất cả thư mục có dạng name* sắp xếp theo thời gian mới nhất
            all_runs = sorted(Path(project).glob(f"{name}*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_runs:
                best_weights = all_runs[0]

    if best_weights and best_weights.exists():
        logger.info(f"Huấn luyện thành công! Trọng số tối ưu lưu tại: {best_weights}")

        # Tùy chọn sao lưu sang thư mục models/ của dự án
        if export_dir is not None:
            dest_dir = Path(export_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            target_pt = dest_dir / "yolov8n_document.pt"
            shutil.copy2(best_weights, target_pt)
            logger.info(f"Đã sao chép model sang: {target_pt}")

            # Đồng thời sao chép sang thư mục models/ ở root repository (nếu khác dest_dir)
            root_models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
            if root_models_dir.resolve() != dest_dir.resolve():
                root_models_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_weights, root_models_dir / "yolov8n_document.pt")
                logger.info(f"Đã sao chép model sang: {root_models_dir / 'yolov8n_document.pt'}")

        return best_weights

    logger.warning("Không tìm thấy file best.pt sau huấn luyện.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n cho nhận diện CCCD & GPLX")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent.parent.parent / "configs" / "data.yaml"),
        help="Đường dẫn file data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", default=None, help="Device: 'cpu' hoặc 'cuda'")
    parser.add_argument("--weights", default="yolov8n.pt", help="Pretrained weights ban đầu")
    parser.add_argument("--project", default="runs/detect", help="Thư mục project run")
    parser.add_argument("--name", default="train_document", help="Tên phiên huấn luyện")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument(
        "--export-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "models"),
        help="Thư mục export model tốt nhất",
    )

    args = parser.parse_args()

    train_yolov8n_document(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        pretrained_weights=args.weights,
        project=args.project,
        name=args.name,
        patience=args.patience,
        export_dir=args.export_dir,
    )

