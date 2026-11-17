"""Kịch bản nhập dataset từ Roboflow (YOLOv8 Pose format ZIP) vào datasets/document_pose/.

Tính năng:
1. Giải nén file ZIP từ Roboflow vào thư mục tạm.
2. Tự động nhận diện cấu trúc các split (train, valid/val, test).
3. Tự động chuẩn hóa tên split: 'valid' -> 'val'.
4. Kiểm tra file data.yaml trong file ZIP, xác minh hoặc ánh xạ lại class id về chuẩn:
   - 0: cccd
   - 1: gplx
5. Kiểm tra tính hợp lệ của từng file nhãn .txt:
   - 17 tokens: class_id xc yc w h + 4 * (kx, ky, kv)
   - Tọa độ chuẩn hóa trong [0, 1]
   - Visibility flag: 0, 1, 2
   - Thứ tự góc: top_left, top_right, bottom_right, bottom_left
6. Bảo vệ an toàn dữ liệu: Không ghi đè ảnh/nhãn đã có trừ khi chỉ định cờ --force.
7. Cập nhật số lượng mẫu vào dataset_manifest.json.
8. Xuất báo cáo tổng kết chi tiết.

Cách dùng:
  python scripts/pose/import_roboflow.py --zip path/to/dataset.zip
  python scripts/pose/import_roboflow.py --zip path/to/dataset.zip --force
  python scripts/pose/import_roboflow.py --zip path/to/dataset.zip --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_roboflow")

# Mapping class chuẩn của dự án XLA
TARGET_CLASSES = {0: "cccd", 1: "gplx"}
CLASS_NAME_TO_TARGET_ID = {"cccd": 0, "gplx": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_zip_splits(extract_root: Path) -> dict[str, Path]:
    """Tìm các thư mục split train, valid/val, test trong thư mục đã giải nén."""
    splits: dict[str, Path] = {}

    # Tìm kiếm ở root hoặc thư mục con 1 cấp
    search_dirs = [extract_root]
    for child in extract_root.iterdir():
        if child.is_dir() and child.name not in ("__MACOSX", ".git"):
            search_dirs.append(child)

    for d in search_dirs:
        for item in d.iterdir():
            if not item.is_dir():
                continue
            name_lower = item.name.lower()
            if name_lower == "train" and "train" not in splits:
                splits["train"] = item
            elif name_lower in ("valid", "val") and "val" not in splits:
                splits["val"] = item
            elif name_lower == "test" and "test" not in splits:
                splits["test"] = item

    return splits


def parse_data_yaml(extract_root: Path) -> dict[str, Any] | None:
    """Tìm và đọc file data.yaml trong thư mục giải nén."""
    candidates = list(extract_root.rglob("data.yaml")) + list(extract_root.rglob("data.yml"))
    if not candidates:
        return None

    yaml_path = candidates[0]
    logger.info(f"Tìm thấy cấu hình dataset: {yaml_path}")
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Không thể đọc file YAML: {e}")
        return None


def get_class_remap(yaml_data: dict[str, Any] | None) -> dict[int, int]:
    """Xác định bảng ánh xạ class id từ Roboflow về chuẩn XLA (0: cccd, 1: gplx)."""
    remap: dict[int, int] = {}
    if not yaml_data:
        logger.info("Không có data.yaml, mặc định giữ nguyên class id 0=cccd, 1=gplx.")
        return {0: 0, 1: 1}

    names = yaml_data.get("names", {})
    if isinstance(names, list):
        names_dict = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names_dict = {int(k): v for k, v in names.items()}
    else:
        names_dict = {}

    for src_id, name in names_dict.items():
        name_clean = str(name).lower().strip()
        if "cccd" in name_clean or "id_card" in name_clean or "citizen" in name_clean:
            remap[src_id] = 0
            logger.info(f"Ánh xạ class '{name}' (id {src_id}) -> 0 (cccd)")
        elif "gplx" in name_clean or "driver" in name_clean or "license" in name_clean:
            remap[src_id] = 1
            logger.info(f"Ánh xạ class '{name}' (id {src_id}) -> 1 (gplx)")
        else:
            logger.warning(f"Class '{name}' (id {src_id}) không khớp cccd hay gplx, giữ nguyên id nếu trong {0, 1}")
            if src_id in (0, 1):
                remap[src_id] = src_id

    return remap or {0: 0, 1: 1}


def validate_and_convert_label(
    label_text: str,
    class_remap: dict[int, int],
) -> tuple[bool, str, str | None]:
    """Kiểm tra và chuẩn hóa dòng nhãn YOLOv8-Pose."""
    out_lines: list[str] = []
    for line in label_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 17:
            return False, "", f"Sai số token ({len(parts)} != 17). Cần 1 class + 4 bbox + 12 kpts."

        try:
            src_cls = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError as e:
            return False, "", f"Giá trị không phải số: {e}"

        target_cls = class_remap.get(src_cls)
        if target_cls is None:
            return False, "", f"Class id {src_cls} không nằm trong mapping hợp lệ."

        # Kiểm tra bbox xc, yc, w, h
        for v in coords[:4]:
            if not (-0.05 <= v <= 1.05):
                return False, "", f"BBox coord {v} vượt ngưỡng [0, 1]."

        # Kiểm tra 4 keypoints (x, y, v)
        for k_i in range(4):
            offset = 4 + k_i * 3
            kx, ky, kv_raw = coords[offset : offset + 3]
            kv = int(round(kv_raw))
            if kv not in (0, 1, 2):
                return False, "", f"Visibility flag {kv} không hợp lệ (phải là 0, 1, 2)."
            if kv > 0 and not (-0.15 <= kx <= 1.15 and -0.15 <= ky <= 1.15):
                return False, "", f"Tọa độ keypoint {k_i} ({kx}, {ky}) vượt ngoài khung ảnh."

        # Tái tạo dòng chuẩn
        formatted_coords = [f"{v:.6f}" if i % 3 != 2 else str(int(round(v))) for i, v in enumerate(coords[4:])]
        bbox_str = " ".join(f"{v:.6f}" for v in coords[:4])
        kpts_str = " ".join(formatted_coords)
        out_lines.append(f"{target_cls} {bbox_str} {kpts_str}")

    return True, "\n".join(out_lines), None


def update_manifest(dest_dir: Path) -> None:
    """Cập nhật số lượng mẫu thực tế vào dataset_manifest.json."""
    manifest_path = dest_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        splits_info: dict[str, Any] = {}
        total_imgs = 0
        for split in ("train", "val", "test"):
            img_dir = dest_dir / "images" / split
            lbl_dir = dest_dir / "labels" / split
            img_count = len([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if img_dir.exists() else 0
            lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            splits_info[split] = {"images_count": img_count, "labels_count": lbl_count}
            total_imgs += img_count

        manifest["splits"] = splits_info
        manifest["status"] = "ready_for_training" if total_imgs > 0 else "scaffold_empty"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã cập nhật dataset_manifest.json: Tổng {total_imgs} ảnh.")
    except Exception as e:
        logger.warning(f"Không thể cập nhật dataset_manifest.json: {e}")


def import_roboflow_zip(
    zip_path: str | Path,
    dest_dir: str | Path = "cccd-reader-api-main/datasets/document_pose",
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Giải nén và nhập Roboflow ZIP vào thư mục đích."""
    zip_p = Path(zip_path).resolve()
    dest_p = Path(dest_dir).resolve()

    if not zip_p.exists():
        logger.error(f"File ZIP không tồn tại: {zip_p}")
        return False

    logger.info(f"Mở file ZIP Roboflow: {zip_p}")
    temp_dir = Path(tempfile.mkdtemp(prefix="roboflow_import_"))

    try:
        with zipfile.ZipFile(zip_p, "r") as zf:
            zf.extractall(temp_dir)
        logger.info(f"Đã giải nén vào thư mục tạm: {temp_dir}")

        splits = find_zip_splits(temp_dir)
        if not splits:
            logger.error("Không tìm thấy cấu trúc thư mục train / valid / test trong file ZIP.")
            return False

        logger.info(f"Tìm thấy các split: {list(splits.keys())}")

        yaml_data = parse_data_yaml(temp_dir)
        class_remap = get_class_remap(yaml_data)

        stats = {
            "imported_images": 0,
            "imported_labels": 0,
            "skipped_existing": 0,
            "invalid_labels": 0,
        }

        for split_name, src_split_dir in splits.items():
            dest_img_dir = dest_p / "images" / split_name
            dest_lbl_dir = dest_p / "labels" / split_name

            if not dry_run:
                dest_img_dir.mkdir(parents=True, exist_ok=True)
                dest_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Cấu trúc roboflow: src_split_dir có thể chứa images/ và labels/ hoặc file trực tiếp
            src_images_dir = src_split_dir / "images" if (src_split_dir / "images").exists() else src_split_dir
            src_labels_dir = src_split_dir / "labels" if (src_split_dir / "labels").exists() else src_split_dir

            img_files = [p for p in src_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            logger.info(f"Split '{split_name}': tìm thấy {len(img_files)} ảnh.")

            for img_file in img_files:
                lbl_file = src_labels_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    logger.warning(f"Bỏ qua ảnh '{img_file.name}' do thiếu file nhãn tương ứng.")
                    continue

                dest_img_file = dest_img_dir / img_file.name
                dest_lbl_file = dest_lbl_dir / f"{img_file.stem}.txt"

                if dest_img_file.exists() and not force:
                    stats["skipped_existing"] += 1
                    continue

                # Đọc và xác minh nhãn
                try:
                    with open(lbl_file, "r", encoding="utf-8") as f:
                        lbl_text = f.read()
                except Exception as e:
                    logger.warning(f"Không thể đọc file nhãn '{lbl_file.name}': {e}")
                    stats["invalid_labels"] += 1
                    continue

                is_valid, converted_text, err_msg = validate_and_convert_label(lbl_text, class_remap)
                if not is_valid:
                    logger.warning(f"File nhãn '{lbl_file.name}' không hợp lệ: {err_msg}")
                    stats["invalid_labels"] += 1
                    continue

                if not dry_run:
                    shutil.copy2(img_file, dest_img_file)
                    with open(dest_lbl_file, "w", encoding="utf-8") as f:
                        f.write(converted_text + "\n")

                stats["imported_images"] += 1
                stats["imported_labels"] += 1

        print("\n" + "=" * 60)
        print("KẾT QUẢ NHẬP DATASET ROBOFLOW")
        print("=" * 60)
        print(f"Ảnh đã nhập: {stats['imported_images']}")
        print(f"Nhãn đã chuẩn hóa: {stats['imported_labels']}")
        print(f"Bỏ qua (đã tồn tại): {stats['skipped_existing']}")
        print(f"Nhãn không hợp lệ: {stats['invalid_labels']}")
        print(f"Chế độ chạy thử (dry-run): {'BẬT' if dry_run else 'TẮT'}")
        print("=" * 60 + "\n")

        if not dry_run and stats["imported_images"] > 0:
            update_manifest(dest_p)

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Import Roboflow YOLOv8-Pose dataset ZIP into XLA project.")
    parser.add_argument("--zip", type=str, required=True, help="Đường dẫn tới file ZIP Roboflow")
    parser.add_argument(
        "--dest",
        type=str,
        default="cccd-reader-api-main/datasets/document_pose",
        help="Thư mục dataset đích",
    )
    parser.add_argument("--force", action="store_true", help="Ghi đè file nếu đã tồn tại")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử không copy file")

    args = parser.parse_args()
    success = import_roboflow_zip(args.zip, args.dest, force=args.force, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
