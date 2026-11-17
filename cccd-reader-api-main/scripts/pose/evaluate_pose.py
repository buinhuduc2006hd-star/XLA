"""Kịch bản đánh giá toàn diện mô hình YOLOv8-Pose trên tập test độc lập.

Định nghĩa và công thức tính các chỉ số đo lường (Metrics):
1. Box Precision & Recall:
   - Precision: Tỷ lệ bounding box dự đoán đúng (IoU >= 0.5) so với tổng số box dự đoán.
   - Recall: Tỷ lệ đối tượng thực tế được phát hiện thành công (IoU >= 0.5).
2. Box mAP (Mean Average Precision):
   - Box mAP50: mAP tại ngưỡng IoU = 0.50.
   - Box mAP50-95: mAP trung bình trên 10 ngưỡng IoU từ 0.50 đến 0.95 (bước nhảy 0.05).
3. Pose mAP (Object Keypoint Similarity - OKS):
   - Pose mAP50: OKS mAP tại ngưỡng 0.50.
   - Pose mAP50-95: OKS mAP trung bình từ 0.50 đến 0.95.
4. PCK@alpha (Percentage of Correct Keypoints):
   - PCK@0.01: Tỷ lệ keypoint có sai số chuẩn hóa (d_kpt / d_diag) <= 0.01 (1% đường chéo thẻ).
   - PCK@0.02: Tỷ lệ keypoint có sai số chuẩn hóa <= 0.02 (2% đường chéo thẻ).
5. Corner Localization Error (CLE):
   - Mean Corner Error (px): Trung bình khoảng cách Euclidean giữa góc dự đoán và góc nhãn (pixels).
   - Median Corner Error (px): Trung vị sai số góc (kháng nhiễu ngoại lai).
   - Mean Normalized Error: Trung bình sai số Euclidean chia cho đường chéo bounding box thẻ.
6. Full-4-Corner Correct Detection Rate:
   - Tỷ lệ thẻ có TẤT CẢ 4 góc đều dự đoán chính xác (mỗi góc có normalized error <= 0.02).
7. Inference Latency (ms):
   - Thời gian xử lý trung bình mỗi ảnh (mili-giây) đo trên phần cứng mục tiêu (CPU/GPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_pose")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_ground_truth_label(label_path: Path, img_w: int, img_h: int) -> list[dict[str, Any]]:
    """Đọc và chuyển đổi nhãn ground truth từ YOLO format sang pixel coordinates."""
    objects = []
    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 17:
                continue
            cls_id = int(parts[0])
            vals = [float(p) for p in parts[1:]]
            xc, yc, w, h = vals[:4]

            bx1 = int((xc - w / 2.0) * img_w)
            by1 = int((yc - h / 2.0) * img_h)
            bx2 = int((xc + w / 2.0) * img_w)
            by2 = int((yc + h / 2.0) * img_h)

            kpts = []
            for k in range(4):
                offset = 4 + k * 3
                kx, ky, kv = vals[offset : offset + 3]
                px = kx * img_w
                py = ky * img_h
                kpts.append((px, py, int(round(kv))))

            diag = math.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)
            objects.append({
                "class_id": cls_id,
                "bbox_xyxy": (bx1, by1, bx2, by2),
                "keypoints": kpts,
                "diag": max(diag, 1.0),
            })
    return objects


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


def evaluate_pose_model(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    device: str = "cpu",
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Chạy đánh giá mô hình YOLOv8-Pose và tính toán toàn bộ các metric."""
    m_path = _resolve_candidate_path(model_path)
    if not m_path.exists():
        logger.error(f"Checkpoint không tồn tại: {m_path}")
        return {"error": f"Model checkpoint not found: {m_path}"}

    yaml_p = _resolve_candidate_path(data_yaml)
    if not yaml_p.exists():
        logger.error(f"File YAML không tồn tại: {yaml_p}")
        return {"error": f"YAML not found: {yaml_p}"}

    logger.info(f"Đang nạp mô hình YOLOv8-Pose: {m_path}")
    from ultralytics import YOLO
    model = YOLO(str(m_path))

    # 1. Chạy Ultralytics Validation chính thức
    logger.info(f"Đang chạy Ultralytics val trên split: {split}...")
    box_p = box_r = box_map50 = box_map50_95 = 0.0
    pose_p = pose_r = pose_map50 = pose_map50_95 = 0.0

    try:
        val_results = model.val(
            data=str(yaml_p),
            split=split,
            device=device,
            conf=conf_threshold,
            plots=False,
            verbose=False,
        )
        if hasattr(val_results, "box"):
            box_p = float(val_results.box.p[0]) if len(val_results.box.p) > 0 else float(val_results.box.mp)
            box_r = float(val_results.box.r[0]) if len(val_results.box.r) > 0 else float(val_results.box.mr)
            box_map50 = float(val_results.box.map50)
            box_map50_95 = float(val_results.box.map)

        if hasattr(val_results, "pose") and val_results.pose is not None:
            pose_p = float(val_results.pose.p[0]) if len(val_results.pose.p) > 0 else float(val_results.pose.mp)
            pose_r = float(val_results.pose.r[0]) if len(val_results.pose.r) > 0 else float(val_results.pose.mr)
            pose_map50 = float(val_results.pose.map50)
            pose_map50_95 = float(val_results.pose.map)
    except Exception as e:
        logger.warning(f"Ultralytics val gặp lỗi (có thể do dataset chưa có ảnh): {e}")

    # 2. Đánh giá chuyên sâu từng ảnh để tính PCK, CLE, và Full 4-Corner Rate
    ds_dir = yaml_p.parent
    img_split_dir = ds_dir / "images" / split
    lbl_split_dir = ds_dir / "labels" / split

    if not img_split_dir.exists():
        img_files = []
    else:
        img_files = [p for p in img_split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    latencies_ms: list[float] = []
    corner_pixel_errors: list[float] = []
    corner_norm_errors: list[float] = []
    total_cards = 0
    detected_cards = 0
    full_4_corners_correct_cards = 0

    pck_001_count = 0  # <= 1% diag
    pck_002_count = 0  # <= 2% diag
    within_5px_count = 0
    within_10px_count = 0
    total_evaluated_keypoints = 0

    logger.info(f"Đang tính toán Keypoint Localization Metrics trên {len(img_files)} ảnh...")

    for img_p in img_files:
        lbl_p = lbl_split_dir / f"{img_p.stem}.txt"
        img_mat = cv2.imread(str(img_p))
        if img_mat is None:
            continue

        h_img, w_img = img_mat.shape[:2]
        gt_objects = parse_ground_truth_label(lbl_p, w_img, h_img)
        total_cards += len(gt_objects)

        t0 = time.perf_counter()
        results = model.predict(source=img_mat, conf=conf_threshold, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        if not results or len(results) == 0:
            continue

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None or len(res.keypoints) == 0:
            continue

        detected_cards += 1
        kpts_xy = res.keypoints.xy.cpu().numpy()
        pred_corners = kpts_xy[0]  # Thẻ dự đoán đầu tiên

        if pred_corners.shape == (4, 2) and gt_objects:
            gt = gt_objects[0]
            gt_kpts = gt["keypoints"]
            card_diag = gt["diag"]

            all_4_corners_ok = True
            for k_idx in range(4):
                gt_x, gt_y, gt_v = gt_kpts[k_idx]
                if gt_v > 0:
                    pred_x, pred_y = pred_corners[k_idx]
                    dist = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                    norm_dist = dist / card_diag

                    corner_pixel_errors.append(dist)
                    corner_norm_errors.append(norm_dist)
                    total_evaluated_keypoints += 1

                    if norm_dist <= 0.01:
                        pck_001_count += 1
                    if norm_dist <= 0.02:
                        pck_002_count += 1
                    else:
                        all_4_corners_ok = False

                    if dist <= 5.0:
                        within_5px_count += 1
                    if dist <= 10.0:
                        within_10px_count += 1
                else:
                    all_4_corners_ok = False

            if all_4_corners_ok and all(kp[2] > 0 for kp in gt_kpts):
                full_4_corners_correct_cards += 1

    # Thống kê tổng hợp
    mean_px_err = float(np.mean(corner_pixel_errors)) if corner_pixel_errors else 0.0
    median_px_err = float(np.median(corner_pixel_errors)) if corner_pixel_errors else 0.0
    mean_norm_err = float(np.mean(corner_norm_errors)) if corner_norm_errors else 0.0
    mean_lat_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    eval_report = {
        "model_path": str(m_path),
        "split": split,
        "device": device,
        "dataset_summary": {
            "total_test_images": len(img_files),
            "total_ground_truth_cards": total_cards,
            "total_detected_cards": detected_cards,
            "card_detection_rate_pct": round((detected_cards / max(1, total_cards)) * 100.0, 2),
        },
        "box_metrics": {
            "precision": round(box_p, 4),
            "recall": round(box_r, 4),
            "mAP50": round(box_map50, 4),
            "mAP50_95": round(box_map50_95, 4),
        },
        "pose_metrics": {
            "precision": round(pose_p, 4),
            "recall": round(pose_r, 4),
            "mAP50": round(pose_map50, 4),
            "mAP50_95": round(pose_map50_95, 4),
        },
        "pck_metrics": {
            "pck_at_0_01_pct": round((pck_001_count / max(1, total_evaluated_keypoints)) * 100.0, 2),
            "pck_at_0_02_pct": round((pck_002_count / max(1, total_evaluated_keypoints)) * 100.0, 2),
            "description": "PCK@alpha là tỷ lệ keypoint có sai số <= alpha * đường chéo thẻ",
        },
        "corner_localization_error": {
            "mean_pixel_error_px": round(mean_px_err, 2),
            "median_pixel_error_px": round(median_px_err, 2),
            "mean_normalized_error": round(mean_norm_err, 4),
            "pct_corners_within_5px": round((within_5px_count / max(1, total_evaluated_keypoints)) * 100.0, 2),
            "pct_corners_within_10px": round((within_10px_count / max(1, total_evaluated_keypoints)) * 100.0, 2),
        },
        "full_four_corners_metric": {
            "correct_cards_count": full_4_corners_correct_cards,
            "full_four_corners_rate_pct": round((full_4_corners_correct_cards / max(1, total_cards)) * 100.0, 2),
            "definition": "Tỷ lệ thẻ mà cả 4 góc đều dự đoán chính xác trong ngưỡng 2% đường chéo thẻ",
        },
        "latency_benchmark": {
            "mean_inference_latency_ms": round(mean_lat_ms, 2),
            "hardware_device": device,
        },
    }
    return eval_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-Pose model on test dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="cccd-reader-api-main/models/document_pose/best.pt",
        help="Path to trained pose best.pt checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cccd-reader-api-main/datasets/document_pose/card_pose.yaml",
        help="Path to card_pose.yaml",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate ('test' or 'val')")
    parser.add_argument("--device", type=str, default="cpu", help="Computing device ('cpu' or '0')")
    parser.add_argument("--json-out", type=str, default=None, help="Save evaluation report to JSON file")
    args = parser.parse_args()

    report = evaluate_pose_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH YOLOV8-POSE")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 60 + "\n")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu kết quả tại: {out_p}")


if __name__ == "__main__":
    main()
