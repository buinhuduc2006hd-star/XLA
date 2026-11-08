"""Script chuẩn bị, kiểm tra toàn vẹn và phân chia dataset YOLOv8n Document Detection.

Sử dụng:
    # 1. Kiểm tra tính toàn vẹn của dataset hiện tại:
    python -m scripts.train.dataset_prepare --check

    # 2. Tự động phân chia từ thư mục raw data gồm ảnh và nhãn txt:
    python -m scripts.train.dataset_prepare --split-raw path/to/raw_folder --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "cccd", 1: "gplx"}


def check_dataset_integrity(dataset_root: Path) -> dict[str, any]:
    """Kiểm tra toàn vẹn cấu trúc dataset YOLO.

    Args:
        dataset_root: Thư mục gốc chứa images/ và labels/ (vd: cccd-reader-api-main/datasets).

    Returns:
        Dict tổng hợp số lượng file, số bounding box theo class, danh sách lỗi nếu có.
    """
    logger.info(f"Đang kiểm tra dataset tại: {dataset_root.resolve()}")

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    stats = {
        "is_valid": True,
        "splits": {},
        "class_counts": Counter(),
        "errors": [],
        "warnings": [],
    }

    if not images_dir.exists() or not labels_dir.exists():
        stats["is_valid"] = False
        stats["errors"].append(f"Không tìm thấy thư mục images/ hoặc labels/ trong {dataset_root}")
        return stats

    for split in ["train", "val", "test"]:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split

        if not split_img_dir.exists():
            stats["warnings"].append(f"Thư mục images/{split} không tồn tại.")
            continue

        images = [f for f in split_img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        labels = [f for f in split_lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"] if split_lbl_dir.exists() else []

        img_stems = {f.stem: f for f in images}
        lbl_stems = {f.stem: f for f in labels}

        # Kiểm tra ảnh không có nhãn
        missing_labels = [stem for stem in img_stems if stem not in lbl_stems]
        if missing_labels:
            stats["warnings"].append(
                f"[{split}] Có {len(missing_labels)} ảnh không có file nhãn .txt tương ứng (vd: {missing_labels[:3]})"
            )

        # Kiểm tra nhãn không có ảnh
        missing_images = [stem for stem in lbl_stems if stem not in img_stems]
        if missing_images:
            stats["warnings"].append(
                f"[{split}] Có {len(missing_images)} file nhãn .txt không có ảnh tương ứng (vd: {missing_images[:3]})"
            )

        # Kiểm tra nội dung các file nhãn
        split_boxes = 0
        for stem, lbl_file in lbl_stems.items():
            if stem not in img_stems:
                continue
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, 1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) != 5:
                    stats["is_valid"] = False
                    stats["errors"].append(
                        f"[{split}] Lỗi định dạng dòng {line_no} file {lbl_file.name}: "
                        f"cần 5 trường (class x y w h), nhận được {len(tokens)}"
                    )
                    continue

                try:
                    c_id = int(tokens[0])
                    xc, yc, w, h = map(float, tokens[1:])
                except ValueError as e:
                    stats["is_valid"] = False
                    stats["errors"].append(
                        f"[{split}] Lỗi parse số dòng {line_no} file {lbl_file.name}: {e}"
                    )
                    continue

                if c_id not in CLASS_NAMES:
                    stats["is_valid"] = False
                    stats["errors"].append(
                        f"[{split}] class_id không hợp lệ ({c_id}) trong {lbl_file.name}. Chỉ chấp nhận 0 (cccd) hoặc 1 (gplx)"
                    )

                # Kiểm tra tọa độ chuẩn hóa [0, 1]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    stats["warnings"].append(
                        f"[{split}] Tọa độ ngoài khoảng [0, 1] ở {lbl_file.name}: ({xc}, {yc}, {w}, {h})"
                    )

                stats["class_counts"][c_id] += 1
                split_boxes += 1

        stats["splits"][split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "box_count": split_boxes,
        }

    return stats


def split_raw_dataset(
    raw_dir: Path,
    target_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Phân chia thư mục ảnh và nhãn raw thành train/val/test theo tỷ lệ chuẩn."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Tổng các tỷ lệ phải bằng 1.0"

    random.seed(seed)
    raw_images = [f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    logger.info(f"Tìm thấy {len(raw_images)} ảnh trong {raw_dir}")

    # Chỉ lấy các ảnh có nhãn .txt tương ứng
    paired_items = []
    for img in raw_images:
        lbl = raw_dir / f"{img.stem}.txt"
        if lbl.exists():
            paired_items.append((img, lbl))
        else:
            logger.warning(f"Bỏ qua ảnh thiếu nhãn: {img.name}")

    random.shuffle(paired_items)
    total = len(paired_items)
    n_train = int(round(total * train_ratio))
    n_val = int(round(total * val_ratio))

    splits = {
        "train": paired_items[:n_train],
        "val": paired_items[n_train : n_train + n_val],
        "test": paired_items[n_train + n_val :],
    }

    for split_name, items in splits.items():
        img_out = target_root / "images" / split_name
        lbl_out = target_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in items:
            shutil.copy2(img_p, img_out / img_p.name)
            shutil.copy2(lbl_p, lbl_out / lbl_p.name)

        logger.info(f"Đã sao chép {len(items)} mẫu vào split: {split_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Dataset Preparation & Integrity Checker")
    parser.add_argument("--check", action="store_true", help="Kiểm tra toàn vẹn dataset")
    parser.add_argument("--dataset-dir", default=None, help="Đường dẫn thư mục dataset")
    parser.add_argument("--split-raw", default=None, help="Đường dẫn thư mục chứa ảnh & nhãn raw để phân chia")
    parser.add_argument("--target-dir", default=None, help="Thư mục đích lưu train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    # Mặc định đường dẫn dataset của project
    default_dataset = Path(__file__).resolve().parent.parent.parent / "datasets"
    dataset_path = Path(args.dataset_dir) if args.dataset_dir else default_dataset

    if args.split_raw:
        target = Path(args.target_dir) if args.target_dir else dataset_path
        split_raw_dataset(
            Path(args.split_raw),
            target,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    if args.check or not args.split_raw:
        report = check_dataset_integrity(dataset_path)
        print("\n=== BÁO CÁO TOÀN VẸN DATASET ===")
        print(f"Hợp lệ: {report['is_valid']}")
        for split, sp_stat in report["splits"].items():
            print(f"- Split '{split}': {sp_stat['image_count']} ảnh, {sp_stat['label_count']} file nhãn, {sp_stat['box_count']} boxes")
        print("Phân bố lớp:")
        for c_id, count in report["class_counts"].items():
            print(f"  * Lớp {c_id} ({CLASS_NAMES.get(c_id, 'unknown')}): {count} boxes")

        if report["warnings"]:
            print(f"\nCảnh báo ({len(report['warnings'])}):")
            for w in report["warnings"][:10]:
                print(f"  [!] {w}")

        if report["errors"]:
            print(f"\nLỗi phát hiện ({len(report['errors'])}):")
            for e in report["errors"][:10]:
                print(f"  [x] {e}")

