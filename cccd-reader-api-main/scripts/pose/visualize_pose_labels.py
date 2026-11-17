"""Kịch bản trực quan hóa (visualize) nhãn 4 góc YOLOv8-Pose trên ảnh thẻ CCCD/GPLX.

Hỗ trợ:
- Vẽ Bounding Box và tên lớp (cccd/gplx).
- Vẽ 4 góc vật lý với màu sắc quy ước chuẩn:
  - Góc 0 (Top-Left):     ĐỎ (Red)
  - Góc 1 (Top-Right):    XANH LÁ (Green)
  - Góc 2 (Bottom-Right): XANH DƯƠNG (Blue)
  - Góc 3 (Bottom-Left):  VÀNG (Yellow)
- Nối 4 góc thành tứ giác để kiểm tra trực quan độ lồi và méo phối cảnh.
- Hiển thị trạng thái visibility (v=0: ẩn, v=1: che khuất/ngoại suy, v=2: rõ nét).
- Chạy trên một ảnh đơn lẻ hoặc xem ngẫu nhiên N ảnh từ một split.

Cách dùng:
  # Xem 1 ảnh cụ thể:
  python scripts/pose/visualize_pose_labels.py --image path/to/img.jpg --label path/to/label.txt --output out.jpg

  # Xem ngẫu nhiên 5 ảnh từ tập train:
  python scripts/pose/visualize_pose_labels.py --split train --num 5 --output-dir debug_visualizations/
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_pose")

CLASS_NAMES = {0: "cccd", 1: "gplx"}
KEYPOINT_COLORS = [
    (0, 0, 255),    # TL: Đỏ (BGR)
    (0, 255, 0),    # TR: Xanh lá (BGR)
    (255, 0, 0),    # BR: Xanh dương (BGR)
    (0, 255, 255),  # BL: Vàng (BGR)
]
KEYPOINT_LABELS = ["0:TL", "1:TR", "2:BR", "3:BL"]


def draw_pose_label(image: np.ndarray, label_text: str) -> np.ndarray:
    """Vẽ bounding box và 4 keypoints lên bản sao của ảnh."""
    vis_img = image.copy()
    h_img, w_img = vis_img.shape[:2]

    for line in label_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 17:
            continue

        cls_id = int(parts[0])
        cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        xc, yc, bw, bh = [float(p) for p in parts[1:5]]

        # Tọa độ bbox pixel
        x1 = int(round((xc - bw / 2.0) * w_img))
        y1 = int(round((yc - bh / 2.0) * h_img))
        x2 = int(round((xc + bw / 2.0) * w_img))
        y2 = int(round((yc + bh / 2.0) * h_img))

        # Vẽ Bounding Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (200, 200, 200), 2)
        cv2.putText(
            vis_img,
            f"Class: {cls_name}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 4 Keypoints
        kpts_pixel: list[tuple[int, int, int]] = []
        for k_i in range(4):
            offset = 4 + k_i * 3
            kx = float(parts[offset])
            ky = float(parts[offset + 1])
            kv = int(round(float(parts[offset + 2])))
            px = int(round(kx * w_img))
            py = int(round(ky * h_img))
            kpts_pixel.append((px, py, kv))

        # Nối 4 góc thành tứ giác
        poly_pts = np.array([[px, py] for px, py, _ in kpts_pixel], dtype=np.int32)
        cv2.polylines(vis_img, [poly_pts], isClosed=True, color=(0, 165, 255), thickness=2)

        # Vẽ từng góc
        for k_i, (px, py, kv) in enumerate(kpts_pixel):
            color = KEYPOINT_COLORS[k_i]
            label = KEYPOINT_LABELS[k_i]

            if kv == 2:
                # Tròn đặc: Rõ nét
                cv2.circle(vis_img, (px, py), 7, color, -1)
                cv2.circle(vis_img, (px, py), 9, (255, 255, 255), 1)
            elif kv == 1:
                # Tròn rỗng: Ngoại suy / che khuất
                cv2.circle(vis_img, (px, py), 7, color, 2)
            else:
                # Dấu X: Ẩn hoàn toàn
                cv2.drawMarker(vis_img, (px, py), color, cv2.MARKER_TILTED_CROSS, 12, 2)

            # Text nhãn góc
            offset_x = 10 if k_i in (1, 2) else -35
            offset_y = -8 if k_i in (0, 1) else 18
            cv2.putText(
                vis_img,
                label,
                (px + offset_x, py + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return vis_img


def visualize_single(image_path: Path, label_path: Path, output_path: Path | None = None) -> None:
    """Trực quan hóa một cặp ảnh - nhãn."""
    if not image_path.exists():
        logger.error(f"File ảnh không tồn tại: {image_path}")
        return
    if not label_path.exists():
        logger.error(f"File nhãn không tồn tại: {label_path}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Không thể giải mã ảnh: {image_path}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lbl_text = f.read()

    vis_img = draw_pose_label(img, lbl_text)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Đã lưu ảnh trực quan hóa: {output_path}")
    else:
        # Nếu không có output, lưu mặc định vào thư mục hiện tại
        def_out = Path(f"vis_{image_path.name}")
        cv2.imwrite(str(def_out), vis_img)
        logger.info(f"Đã lưu ảnh trực quan hóa vào: {def_out}")


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv8-Pose keypoint labels on Vietnamese ID cards.")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh")
    parser.add_argument("--label", type=str, default=None, help="Đường dẫn file nhãn .txt")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file ảnh kết quả")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="cccd-reader-api-main/datasets/document_pose",
        help="Thư mục gốc dataset pose",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Tập dữ liệu để lấy mẫu")
    parser.add_argument("--num", type=int, default=3, help="Số ảnh lấy mẫu khi không chỉ định file cụ thể")
    parser.add_argument("--output-dir", type=str, default="debug_visualizations", help="Thư mục lưu ảnh hàng loạt")

    args = parser.parse_args()

    if args.image and args.label:
        out_p = Path(args.output) if args.output else None
        visualize_single(Path(args.image), Path(args.label), out_p)
    else:
        # Lấy mẫu ngẫu nhiên từ split
        ds_dir = Path(args.data_dir).resolve()
        img_dir = ds_dir / "images" / args.split
        lbl_dir = ds_dir / "labels" / args.split

        if not img_dir.exists() or not lbl_dir.exists():
            logger.warning(f"Thư mục split {args.split} chưa có dữ liệu tại {ds_dir}")
            sys.exit(0)

        img_files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}]
        if not img_files:
            logger.info(f"Tập {args.split} chưa có ảnh nào để trực quan hóa (Dataset scaffold rỗng).")
            sys.exit(0)

        sampled = random.sample(img_files, min(args.num, len(img_files)))
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_p in sampled:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                out_file = out_dir / f"vis_{img_p.name}"
                visualize_single(img_p, lbl_p, out_file)


if __name__ == "__main__":
    main()
