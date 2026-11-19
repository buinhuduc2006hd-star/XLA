"""Kịch bản nhập dataset từ Roboflow (YOLOv8 Pose format ZIP) vào datasets/document_pose/.

Quy chuẩn Keypoint Roboflow (XLA Specification):
  - Index 0: point_up_left (Top-Left corner)
  - Index 1: point_up_right (Top-Right corner)
  - Index 2: point_bottom_right (Bottom-Right corner)
  - Index 3: point_bottom_left (Bottom-Left corner)

Quy chuẩn Document Classes:
  - Class 0: cccd
  - Class 1: gplx

Quy trình nhập dữ liệu:
1. Giải nén file ZIP từ Roboflow vào thư mục tạm.
2. Đọc metadata thực tế từ data.yaml (hoặc data.yml):
   - Kiểm tra kpt_shape (bắt buộc 4 keypoints: [4, 2] hoặc [4, 3]).
   - Phân tích và xác minh danh sách tên keypoint nguồn.
   - Xác định bảng ánh xạ keypoint index (kpt_remap) nếu thứ tự nguồn khác thứ tự đích.
   - Báo lỗi rõ ràng nếu metadata mâu thuẫn hoặc không thể xác minh mapping.
3. Kiểm tra class mapping (cccd -> 0, gplx -> 1).
4. Chuẩn hóa tên split (valid -> val, train, test).
5. Kiểm tra tính hợp lệ từng dòng nhãn .txt và thực hiện remap keypoint index (nếu cần).
6. Bảo vệ an toàn dữ liệu: Không ghi đè ảnh/nhãn đã có trừ khi chỉ định cờ --force.
7. Cập nhật số lượng mẫu vào dataset_manifest.json.
8. Hỗ trợ cờ --dry-run để chạy kiểm tra mapping mà không ghi file.

Cách dùng:
  python scripts/pose/import_roboflow.py --zip path/to/roboflow_dataset.zip
  python scripts/pose/import_roboflow.py --zip path/to/roboflow_dataset.zip --dry-run
  python scripts/pose/import_roboflow.py --zip path/to/roboflow_dataset.zip --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_roboflow")

# Quy chuẩn Roboflow Keypoint Specification của XLA
TARGET_KEYPOINTS = [
    "point_up_left",       # Index 0: Top Left
    "point_up_right",      # Index 1: Top Right
    "point_bottom_right",  # Index 2: Bottom Right
    "point_bottom_left",   # Index 3: Bottom Left
]

# Mapping class chuẩn của dự án XLA
TARGET_CLASSES = {0: "cccd", 1: "gplx"}
CLASS_NAME_TO_TARGET_ID = {"cccd": 0, "gplx": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_candidate_path(raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    cwd = Path.cwd()
    if (cwd / p).exists():
        return (cwd / p).resolve()
    if "cccd-reader-api-main" in cwd.parts:
        idx = cwd.parts.index("cccd-reader-api-main")
        root = Path(*cwd.parts[:idx])
        if (root / p).exists():
            return (root / p).resolve()
    if (cwd / "cccd-reader-api-main" / p).exists():
        return (cwd / "cccd-reader-api-main" / p).resolve()
    return p.resolve()


def find_dataset_splits(extract_root: Path) -> dict[str, Path]:
    """Tự động tìm kiếm thư mục train/val/test trong thư mục giải nén."""
    splits: dict[str, Path] = {}
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
    logger.info(f"Tìm thấy cấu hình dataset Roboflow: {yaml_path}")
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


def normalize_kpt_name(name: str) -> str:
    """Chuẩn hóa tên keypoint để so khớp."""
    s = str(name).lower().strip()
    s = re.sub(r"[\s\-_]+", "_", s)
    return s


def identify_target_keypoint_index(name: str) -> int | None:
    """Xác định index logic mục tiêu (0..3) từ tên keypoint."""
    clean = normalize_kpt_name(name)
    # 0: point_up_left / Top-Left
    if any(k in clean for k in ["point_up_left", "up_left", "top_left", "tl", "corner_tl"]):
        return 0
    # 1: point_up_right / Top-Right
    if any(k in clean for k in ["point_up_right", "up_right", "top_right", "tr", "corner_tr"]):
        return 1
    # 2: point_bottom_right / Bottom-Right
    if any(k in clean for k in ["point_bottom_right", "bottom_right", "down_right", "br", "corner_br"]):
        return 2
    # 3: point_bottom_left / Bottom-Left
    if any(k in clean for k in ["point_bottom_left", "bottom_left", "down_left", "bl", "corner_bl"]):
        return 3
    return None


def get_keypoint_remap(yaml_data: dict[str, Any] | None) -> dict[int, int]:
    """Kiểm tra và xác định bảng remap thứ tự Keypoint từ Roboflow về thứ tự chuẩn XLA.

    Target index:
      0: point_up_left (TL)
      1: point_up_right (TR)
      2: point_bottom_right (BR)
      3: point_bottom_left (BL)

    Returns:
      dict ánh xạ: src_kpt_idx -> target_kpt_idx.
    """
    if not yaml_data:
        logger.info("Không có data.yaml để kiểm tra keypoints. Áp dụng thứ tự mặc định 0..3.")
        return {0: 0, 1: 1, 2: 2, 3: 3}

    # Kiểm tra kpt_shape nếu có
    kpt_shape = yaml_data.get("kpt_shape")
    if kpt_shape:
        num_kpts = kpt_shape[0] if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) > 0 else None
        if num_kpts is not None and num_kpts != 4:
            raise ValueError(f"kpt_shape không hợp lệ: {kpt_shape}. XLA yêu cầu chính xác 4 keypoints (4 góc).")

    # Tìm danh sách tên keypoint trong metadata
    kpt_names = None
    for field in ["keypoints", "kpt_names", "names_kpt", "keypoint_names"]:
        if field in yaml_data and isinstance(yaml_data[field], list):
            kpt_names = yaml_data[field]
            break

    if not kpt_names:
        logger.info("data.yaml không khai báo danh sách 'keypoints'. Giả định thứ tự chuẩn 0=point_up_left, 1=point_up_right, 2=point_bottom_right, 3=point_bottom_left.")
        return {0: 0, 1: 1, 2: 2, 3: 3}

    if len(kpt_names) != 4:
        raise ValueError(f"Số lượng keypoint khai báo trong metadata ({len(kpt_names)}) khác 4: {kpt_names}")

    remap: dict[int, int] = {}
    used_targets: set[int] = set()

    for src_idx, name in enumerate(kpt_names):
        target_idx = identify_target_keypoint_index(str(name))
        if target_idx is None:
            raise ValueError(
                f"Không thể nhận diện tên keypoint '{name}' tại index {src_idx}. "
                f"Tên hợp lệ phải khớp một trong: {TARGET_KEYPOINTS}"
            )
        if target_idx in used_targets:
            raise ValueError(
                f"Trùng lặp mục tiêu mapping: keypoint '{name}' tại index {src_idx} "
                f"được ánh xạ về target index {target_idx} đã được sử dụng."
            )
        remap[src_idx] = target_idx
        used_targets.add(target_idx)

    if set(remap.values()) != {0, 1, 2, 3}:
        raise ValueError(f"Mapping keypoint không bao phủ đủ 4 góc 0..3: {remap}")

    if remap == {0: 0, 1: 1, 2: 2, 3: 3}:
        logger.info(f"Thứ tự Keypoint nguồn trùng khớp 100% với chuẩn XLA: {kpt_names}")
    else:
        logger.info(f"Phát hiện thứ tự Keypoint nguồn khác chuẩn XLA: {kpt_names} -> Remap mapping: {remap}")

    return remap


def validate_and_convert_label(
    label_text: str,
    class_remap: dict[int, int],
    keypoint_remap: dict[int, int],
) -> tuple[bool, str, str | None]:
    """Kiểm tra, chuyển đổi class và remap thứ tự keypoints trong từng dòng nhãn."""
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

        # Bbox xc, yc, w, h
        bbox_coords = coords[:4]
        for v in bbox_coords:
            if not (-0.05 <= v <= 1.05):
                return False, "", f"BBox coord {v} vượt ngưỡng [0, 1]."

        # Thu thập 4 keypoints nguồn (kx, ky, kv)
        src_kpts = []
        for k_i in range(4):
            kx = coords[4 + k_i * 3]
            ky = coords[4 + k_i * 3 + 1]
            kv = int(round(coords[4 + k_i * 3 + 2]))
            if kv not in (0, 1, 2):
                return False, "", f"Visibility flag {kv} không hợp lệ (phải là 0, 1 hoặc 2)."
            src_kpts.append((kx, ky, kv))

        # Remap keypoints sang target order
        target_kpts = [None] * 4
        for src_i, (kx, ky, kv) in enumerate(src_kpts):
            tgt_i = keypoint_remap[src_i]
            target_kpts[tgt_i] = (kx, ky, kv)

        # Định dạng lại dòng nhãn đầu ra
        line_tokens = [str(target_cls)] + [f"{c:.6f}" for c in bbox_coords]
        for kx, ky, kv in target_kpts:
            line_tokens.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

        out_lines.append(" ".join(line_tokens))

    return True, "\n".join(out_lines), None


def import_roboflow_zip(
    zip_path: str | Path,
    target_pose_dir: str | Path = "datasets/document_pose",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Nhập dataset từ Roboflow Pose ZIP vào thư mục datasets/document_pose."""
    resolved_zip = _resolve_candidate_path(zip_path)
    if not resolved_zip.exists():
        raise FileNotFoundError(f"Không tìm thấy file ZIP: {resolved_zip}")

    target_dir = _resolve_candidate_path(target_pose_dir)
    logger.info(f"Bắt đầu import từ: {resolved_zip}")
    logger.info(f"Thư mục đích: {target_dir} (Dry-run: {dry_run}, Force: {force})")

    report: dict[str, Any] = {
        "zip_path": str(resolved_zip),
        "target_dir": str(target_dir),
        "dry_run": dry_run,
        "splits_processed": {},
        "total_images": 0,
        "total_labels": 0,
        "keypoint_remap_applied": None,
        "class_remap_applied": None,
        "errors": [],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        logger.info(f"Giải nén file ZIP vào thư mục tạm: {tmp_path}")
        with zipfile.ZipFile(resolved_zip, "r") as zf:
            zf.extractall(tmp_path)

        # 1. Đọc data.yaml
        yaml_data = parse_data_yaml(tmp_path)
        class_remap = get_class_remap(yaml_data)
        keypoint_remap = get_keypoint_remap(yaml_data)

        report["class_remap_applied"] = class_remap
        report["keypoint_remap_applied"] = keypoint_remap

        # 2. Tìm các split
        splits = find_dataset_splits(tmp_path)
        if not splits:
            raise ValueError(f"Không tìm thấy cấu trúc split train/val/test nào trong {resolved_zip}")

        logger.info(f"Tìm thấy các splits: {list(splits.keys())}")

        # 3. Xử lý từng split
        for split_name, split_path in splits.items():
            img_dir = split_path / "images" if (split_path / "images").exists() else split_path
            lbl_dir = split_path / "labels" if (split_path / "labels").exists() else split_path

            target_img_dir = target_dir / "images" / split_name
            target_lbl_dir = target_dir / "labels" / split_name

            if not dry_run:
                target_img_dir.mkdir(parents=True, exist_ok=True)
                target_lbl_dir.mkdir(parents=True, exist_ok=True)

            img_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            split_stats = {"images_found": len(img_files), "images_copied": 0, "labels_copied": 0, "labels_invalid": 0}

            for img_file in img_files:
                stem = img_file.stem
                lbl_candidates = list(lbl_dir.glob(f"{stem}.txt"))

                if not lbl_candidates:
                    report["errors"].append(f"Split {split_name}: Ảnh '{img_file.name}' không có file nhãn .txt tương ứng.")
                    split_stats["labels_invalid"] += 1
                    continue

                lbl_file = lbl_candidates[0]
                lbl_text = lbl_file.read_text(encoding="utf-8", errors="replace")

                is_valid, converted_lbl, err_msg = validate_and_convert_label(lbl_text, class_remap, keypoint_remap)
                if not is_valid:
                    report["errors"].append(f"Split {split_name}, file '{lbl_file.name}': {err_msg}")
                    split_stats["labels_invalid"] += 1
                    continue

                dst_img = target_img_dir / img_file.name
                dst_lbl = target_lbl_dir / f"{stem}.txt"

                if not force and dst_img.exists():
                    logger.debug(f"Ảnh {dst_img.name} đã tồn tại trong target, bỏ qua (không ghi đè).")
                    continue

                if not dry_run:
                    shutil.copy2(img_file, dst_img)
                    dst_lbl.write_text(converted_lbl, encoding="utf-8")

                split_stats["images_copied"] += 1
                split_stats["labels_copied"] += 1

            report["splits_processed"][split_name] = split_stats
            report["total_images"] += split_stats["images_copied"]
            report["total_labels"] += split_stats["labels_copied"]

        # 4. Cập nhật manifest nếu không phải dry-run
        if not dry_run:
            manifest_path = target_dir / "dataset_manifest.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    for s_name in ["train", "val", "test"]:
                        cur_imgs = len(list((target_dir / "images" / s_name).glob("*.*"))) if (target_dir / "images" / s_name).exists() else 0
                        cur_lbls = len(list((target_dir / "labels" / s_name).glob("*.txt"))) if (target_dir / "labels" / s_name).exists() else 0
                        manifest_data.setdefault("splits", {}).setdefault(s_name, {})["images_count"] = cur_imgs
                        manifest_data.setdefault("splits", {}).setdefault(s_name, {})["labels_count"] = cur_lbls
                    manifest_data["status"] = "dataset_ready" if manifest_data["splits"]["train"]["images_count"] > 0 else "scaffold_initialized"
                    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
                    logger.info("Đã cập nhật số lượng mẫu vào dataset_manifest.json.")
                except Exception as e:
                    logger.warning(f"Không thể cập nhật dataset_manifest.json: {e}")

    logger.info("=== BÁO CÁO NHẬP DATASET ROBOFLOW ===")
    logger.info(f"Tổng ảnh hợp lệ: {report['total_images']}")
    logger.info(f"Tổng nhãn hợp lệ: {report['total_labels']}")
    logger.info(f"Keypoint remap applied: {report['keypoint_remap_applied']}")
    if report["errors"]:
        logger.warning(f"Có {len(report['errors'])} cảnh báo / lỗi (xem chi tiết trong report).")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Nhập dataset Roboflow YOLOv8-Pose vào datasets/document_pose/")
    parser.add_argument("--zip", required=True, help="Đường dẫn đến file ZIP từ Roboflow")
    parser.add_argument("--target", default="datasets/document_pose", help="Thư mục đích (mặc định datasets/document_pose)")
    parser.add_argument("--force", action="store_true", help="Ghi đè file nếu đã tồn tại")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra và xuất báo cáo, không ghi đĩa")
    args = parser.parse_args()

    try:
        report = import_roboflow_zip(
            zip_path=args.zip,
            target_pose_dir=args.target,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Lỗi khi import dataset: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
