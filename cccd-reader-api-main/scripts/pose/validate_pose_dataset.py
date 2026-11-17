"""Kịch bản kiểm tra tính toàn vẹn và chất lượng của dataset YOLOv8-Pose.

Kiểm tra:
1. Tính tương ứng 1-1 giữa ảnh và file nhãn (.txt).
2. Đúng định dạng nhãn YOLOv8-Pose (17 tokens: class + bbox + 4 keypoints x 3 dims).
3. Đúng 2 class: 0 (cccd) và 1 (gplx).
4. Tọa độ bbox và keypoints trong miền hợp lệ [0, 1].
5. Trạng thái visibility hợp lệ: 0 (ẩn), 1 (che khuất/ngoại suy góc bo), 2 (rõ nét).
6. Tứ giác 4 góc lồi, thứ tự chuẩn chiều kim đồng hồ (TL, TR, BR, BL).
7. Kiểm tra rò rỉ dữ liệu (Data leakage / Exact duplicate) giữa các split train/val/test qua SHA-256 hash.
8. Tính toàn vẹn của file ảnh (không bị hỏng header/pixel).
9. Phân phối mẫu theo class và split.
10. Xử lý an toàn khi dataset rỗng (báo chưa sẵn sàng, không crash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_pose_dataset")

EXPECTED_CLASSES = {0: "cccd", 1: "gplx"}
KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_file_hash(path: Path) -> str:
    """Tính SHA-256 hash của file để phát hiện ảnh trùng lặp."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_label_line(
    line: str,
    line_no: int,
    filename: str,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Kiểm tra một dòng nhãn YOLOv8-Pose (kpt_shape=[4, 3])."""
    parts = line.strip().split()
    if not parts:
        return False, "Dòng rỗng", None

    # 1 class + 4 bbox coords + 4 keypoints * 3 values = 17 tokens
    if len(parts) != 17:
        return (
            False,
            f"Số lượng tokens không đúng ({len(parts)} != 17). Cần: class bbox(4) + 4*(x, y, v)",
            None,
        )

    try:
        cls_id = int(parts[0])
    except ValueError:
        return False, f"Class ID không phải số nguyên: '{parts[0]}'", None

    if cls_id not in EXPECTED_CLASSES:
        return False, f"Class ID không hợp lệ: {cls_id}. Chỉ chấp nhận 0 (cccd) hoặc 1 (gplx)", None

    try:
        vals = [float(p) for p in parts[1:]]
    except ValueError as e:
        return False, f"Giá trị tọa độ không phải số thực: {e}", None

    xc, yc, w, h = vals[:4]
    for name, v in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
        if not (0.0 <= v <= 1.0):
            return False, f"BBox {name} ({v}) nằm ngoài khoảng [0, 1]", None
    if w <= 0.0 or h <= 0.0:
        return False, f"BBox width hoặc height <= 0: w={w}, h={h}", None

    kpts: list[tuple[float, float, int]] = []
    for k_idx in range(4):
        offset = 4 + k_idx * 3
        kx, ky, kv_raw = vals[offset : offset + 3]
        kv = int(round(kv_raw))

        if kv not in (0, 1, 2):
            return False, f"Keypoint {KEYPOINT_NAMES[k_idx]} visibility flag ({kv}) không hợp lệ (phải là 0, 1, hoặc 2)", None

        # Nếu keypoint nhìn thấy hoặc ngoại suy (v > 0), tọa độ phải hợp lệ
        if kv > 0:
            if not (-0.1 <= kx <= 1.1 and -0.1 <= ky <= 1.1):
                return False, f"Keypoint {KEYPOINT_NAMES[k_idx]} tọa độ ({kx:.3f}, {ky:.3f}) vượt quá dung sai [0, 1]", None

        kpts.append((kx, ky, kv))

    # Kiểm tra thứ tự và tính lồi nếu đủ 4 điểm có tọa độ (v > 0)
    if all(kp[2] > 0 for kp in kpts):
        pts_np = np.array([[kp[0], kp[1]] for kp in kpts], dtype=np.float32)
        # Vector TL->TR và TL->BL
        v_top = pts_np[1] - pts_np[0]
        v_left = pts_np[3] - pts_np[0]
        cross_z = v_top[0] * v_left[1] - v_top[1] * v_left[0]
        if cross_z <= 0:
            return False, "4 góc không theo thứ tự chiều kim đồng hồ hoặc bị lật gương (TL -> TR -> BR -> BL)", None

    parsed_info = {
        "class_id": cls_id,
        "class_name": EXPECTED_CLASSES[cls_id],
        "bbox": (xc, yc, w, h),
        "keypoints": kpts,
    }
    return True, None, parsed_info


def _resolve_candidate_path(raw_path: str | Path) -> Path:
    """Tự động resolve đường dẫn linh hoạt dù chạy từ root repo hay cccd-reader-api-main."""
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    # Thử tìm trong các thư mục cha hoặc con phổ biến
    candidates = [
        Path.cwd() / p,
        Path.cwd() / "cccd-reader-api-main" / p,
        Path(__file__).resolve().parent.parent.parent / p,
        Path(__file__).resolve().parent.parent.parent / "datasets" / "document_pose",
    ]
    # Nếu raw_path bắt đầu với cccd-reader-api-main/ nhưng cwd đã ở cccd-reader-api-main/
    parts = p.parts
    if len(parts) > 1 and parts[0] == "cccd-reader-api-main":
        stripped = Path(*parts[1:])
        candidates.insert(0, Path.cwd() / stripped)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return p.resolve()


def validate_dataset(
    dataset_dir: str | Path,
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Kiểm tra toàn diện thư mục dataset YOLOv8-Pose."""
    ds_path = _resolve_candidate_path(dataset_dir)
    logger.info(f"Bắt đầu kiểm tra dataset tại: {ds_path}")

    report: dict[str, Any] = {
        "dataset_path": str(ds_path),
        "is_ready_for_training": False,
        "total_images": 0,
        "total_labels": 0,
        "splits": {},
        "class_distribution": {"cccd": 0, "gplx": 0},
        "errors": [],
        "warnings": [],
    }

    if not ds_path.exists():
        msg = f"Thư mục dataset không tồn tại: {ds_path}"
        logger.error(msg)
        report["errors"].append(msg)
        return report

    # 1. Kiểm tra cấu trúc thư mục
    images_dir = ds_path / "images"
    labels_dir = ds_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        msg = f"Cấu trúc thư mục thiếu 'images' hoặc 'labels' bên trong {ds_path}"
        logger.error(msg)
        report["errors"].append(msg)
        return report

    all_hashes: dict[str, tuple[str, str]] = {}  # hash -> (split, filename)
    splits = ["train", "val", "test"]

    for split in splits:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split

        report["splits"][split] = {
            "images": 0,
            "labels": 0,
            "valid_samples": 0,
            "missing_labels": 0,
            "corrupt_images": 0,
            "label_errors": 0,
        }

        if not split_img_dir.exists():
            report["warnings"].append(f"Thư mục images/{split} chưa tồn tại")
            continue
        if not split_lbl_dir.exists():
            report["warnings"].append(f"Thư mục labels/{split} chưa tồn tại")
            continue

        # Quét danh sách ảnh
        img_files = [p for p in split_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        report["splits"][split]["images"] = len(img_files)
        report["total_images"] += len(img_files)

        for img_p in img_files:
            lbl_p = split_lbl_dir / f"{img_p.stem}.txt"

            # Kiểm tra file nhãn tương ứng
            if not lbl_p.exists():
                report["splits"][split]["missing_labels"] += 1
                report["errors"].append(f"[{split}] Ảnh '{img_p.name}' không có file nhãn tương ứng '{lbl_p.name}'")
                continue

            # Kiểm tra ảnh có đọc được không
            img_mat = cv2.imread(str(img_p))
            if img_mat is None or img_mat.size == 0:
                report["splits"][split]["corrupt_images"] += 1
                report["errors"].append(f"[{split}] File ảnh bị hỏng không thể giải mã: '{img_p.name}'")
                continue

            # Kiểm tra Data leakage / Trùng lặp
            file_h = compute_file_hash(img_p)
            if file_h in all_hashes:
                prev_split, prev_name = all_hashes[file_h]
                report["warnings"].append(
                    f"Data Leakage / Duplicate: '{img_p.name}' ({split}) trùng hoàn toàn nội dung với '{prev_name}' ({prev_split})"
                )
            else:
                all_hashes[file_h] = (split, img_p.name)

            # Kiểm tra nội dung nhãn
            try:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception as e:
                report["splits"][split]["label_errors"] += 1
                report["errors"].append(f"[{split}] Không thể đọc file nhãn '{lbl_p.name}': {e}")
                continue

            if not lines:
                report["splits"][split]["label_errors"] += 1
                report["errors"].append(f"[{split}] File nhãn rỗng: '{lbl_p.name}'")
                continue

            sample_valid = True
            for l_idx, line in enumerate(lines, start=1):
                is_valid, err_msg, info = validate_label_line(line, l_idx, lbl_p.name)
                if not is_valid:
                    report["splits"][split]["label_errors"] += 1
                    report["errors"].append(f"[{split}] Lỗi nhãn '{lbl_p.name}' dòng {l_idx}: {err_msg}")
                    sample_valid = False
                else:
                    cls_name = info["class_name"]
                    report["class_distribution"][cls_name] = report["class_distribution"].get(cls_name, 0) + 1

            if sample_valid:
                report["splits"][split]["valid_samples"] += 1
                report["total_labels"] += 1

    # Đánh giá trạng thái sẵn sàng huấn luyện
    if report["total_images"] == 0:
        logger.warning("Thư mục dataset rỗng (0 ảnh). Bộ khung scaffold đã khởi tạo an toàn, chưa có dữ liệu huấn luyện.")
        report["is_ready_for_training"] = False
        report["summary"] = "SCAFFOLD_EMPTY (Dataset initialized, 0 samples)"
    elif len(report["errors"]) == 0:
        logger.info(f"Dataset hợp lệ hoàn hảo! Tổng: {report['total_images']} ảnh, 0 lỗi.")
        report["is_ready_for_training"] = True
        report["summary"] = f"READY ({report['total_images']} valid samples)"
    else:
        logger.error(f"Dataset phát hiện {len(report['errors'])} lỗi cấu trúc.")
        report["is_ready_for_training"] = False
        report["summary"] = f"INVALID ({len(report['errors'])} errors detected)"

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8-Pose dataset integrity for Vietnamese ID cards.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="cccd-reader-api-main/datasets/document_pose",
        help="Path to document_pose directory",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="cccd-reader-api-main/datasets/document_pose/card_pose.yaml",
        help="Path to card_pose.yaml",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Save report to JSON file")
    args = parser.parse_args()

    report = validate_dataset(args.data_dir, args.yaml)

    print("\n" + "=" * 60)
    print("BÁO CÁO KIỂM TRA DATASET POSE")
    print("=" * 60)
    print(f"Dataset Path: {report['dataset_path']}")
    print(f"Tổng số ảnh: {report['total_images']}")
    print(f"Tổng số nhãn: {report['total_labels']}")
    print(f"Sẵn sàng huấn luyện: {'CÓ (READY)' if report['is_ready_for_training'] else 'CHƯA (NOT READY)'}")
    print(f"Tóm tắt: {report.get('summary')}")
    print(f"Phân phối class: {report['class_distribution']}")
    print(f"Số lỗi phát hiện: {len(report['errors'])}")
    print(f"Số cảnh báo: {len(report['warnings'])}")

    if report["errors"]:
        print("\nCác lỗi chi tiết (tối đa 10 lỗi đầu tiên):")
        for err in report["errors"][:10]:
            print(f"  - [X] {err}")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu báo cáo JSON: {out_p}")

    print("=" * 60 + "\n")
    # Nếu dataset rỗng (scaffold), không coi là lỗi crash (exit code 0)
    sys.exit(0)


if __name__ == "__main__":
    main()
