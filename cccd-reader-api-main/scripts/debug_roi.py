"""Công cụ debug và trực quan hóa ROI (Region of Interest) cho CCCD / GPLX.

Tính năng:
- Đọc ảnh mẫu giấy tờ.
- Chuẩn hóa kích thước tương tự pipeline trích xuất.
- Vẽ tất cả các bounding box ROI lên ảnh và gắn nhãn tên trường.
- Cắt và lưu riêng ảnh từng trường để kiểm tra vị trí ROI.
- Cho phép kiểm tra và in bảng thông số tọa độ (relative & pixel).
- Hỗ trợ truyền tùy chỉnh tọa độ (override) từ dòng lệnh hoặc file.

Sử dụng:
    python scripts/debug_roi.py <duong_dan_anh> [--output-dir <thu_muc_luu>] [--card-type cccd|gplx]

LƯU Ý BẢO MẬT:
    Không lưu hoặc commit ảnh chứa thông tin cá nhân thật lên git repository!
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

# Cấu hình ROI mặc định cho CCCD (Mẫu CCCD chip 2021 Thông tư 06/2021/TT-BCA)
CCCD_ROIS: dict[str, tuple[float, float, float, float]] = {
    "id_number": (0.42, 0.42, 0.88, 0.52),
    "full_name": (0.27, 0.53, 0.61, 0.63),
    "date_of_birth": (0.56, 0.61, 0.76, 0.70),
    "gender": (0.46, 0.665, 0.545, 0.74),
    "nationality": (0.81, 0.645, 0.985, 0.735),
    "place_of_origin": (0.28, 0.79, 0.82, 0.855),
    "residence_line_1": (0.665, 0.855, 0.93, 0.92),
    "residence_line_2": (0.28, 0.91, 0.84, 0.985),
    "expiry_date": (0.04, 0.78, 0.32, 0.92),
}

# Cấu hình ROI cho GPLX
GPLX_ROIS: dict[str, tuple[float, float, float, float]] = {
    "license_number": (0.61, 0.235, 0.86, 0.315),
    "full_name": (0.62, 0.315, 0.85, 0.380),
    "date_of_birth": (0.675, 0.385, 0.84, 0.455),
    "nationality": (0.675, 0.455, 0.85, 0.525),
    "address": (0.38, 0.580, 0.91, 0.665),
    "license_class": (0.165, 0.795, 0.245, 0.865),
    "expiry": (0.28, 0.895, 0.55, 0.965),
}

STANDARD_WIDTH = 1061
BOX_PADDING = 0.008

# Bảng màu hiển thị (BGR)
COLORS = [
    (0, 0, 255),    # Đỏ
    (0, 165, 255),  # Cam
    (0, 255, 255),  # Vàng
    (0, 255, 0),    # Xanh lá
    (255, 255, 0),  # Cyan
    (255, 0, 0),    # Xanh dương
    (255, 0, 255),  # Tím
    (128, 0, 128),  # Tím sẫm
    (0, 128, 255),  # Hổ phách
]


def normalize_image(image: np.ndarray, target_width: int = STANDARD_WIDTH) -> np.ndarray:
    """Chuẩn hóa chiều rộng ảnh về kích thước cố định."""
    height, width = image.shape[:2]
    if width == target_width:
        return image
    scale = target_width / width
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)


def get_crop_coords(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
    padding: float = BOX_PADDING,
) -> tuple[int, int, int, int]:
    """Tính toán tọa độ pixel (left, top, right, bottom) từ tỷ lệ."""
    x1, y1, x2, y2 = box
    left = max(int(width * (x1 - padding)), 0)
    right = min(int(width * (x2 + padding)), width)
    top = max(int(height * (y1 - padding)), 0)
    bottom = min(int(height * (y2 + padding)), height)
    return left, top, right, bottom


def debug_rois(
    image_path: str | Path,
    output_dir: str | Path = "debug_output",
    card_type: str = "cccd",
    custom_rois: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, dict]:
    """Vẽ ROI và xuất ảnh crop từng trường."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")

    norm_img = normalize_image(image)
    height, width = norm_img.shape[:2]

    out_path = Path(output_dir)
    crops_dir = out_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rois = custom_rois or (CCCD_ROIS if card_type == "cccd" else GPLX_ROIS)
    annotated = norm_img.copy()

    summary: dict[str, dict] = {}
    print(f"\n=== DEBUG ROI CHO: {card_type.upper()} ({width}x{height}px) ===")
    print(f"{'Trường':<20} | {'Tọa độ rel (x1, y1, x2, y2)':<32} | {'Tọa độ pixel (L, T, R, B)':<26}")
    print("-" * 86)

    for idx, (field_name, box) in enumerate(rois.items()):
        color = COLORS[idx % len(COLORS)]
        left, top, right, bottom = get_crop_coords(width, height, box)

        # Cắt và lưu riêng ảnh crop của trường
        crop = norm_img[top:bottom, left:right]
        crop_file = crops_dir / f"{field_name}.png"
        if crop.size > 0:
            cv2.imwrite(str(crop_file), crop)

        # Vẽ lên ảnh annotated
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"{field_name}"
        cv2.putText(
            annotated,
            label,
            (left, max(top - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        box_rel_str = f"({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f})"
        box_px_str = f"({left}, {top}, {right}, {bottom})"
        print(f"{field_name:<20} | {box_rel_str:<32} | {box_px_str:<26}")

        summary[field_name] = {
            "relative_box": box,
            "pixel_box": (left, top, right, bottom),
            "crop_path": str(crop_file),
            "size": (right - left, bottom - top),
        }

    annotated_file = out_path / f"annotated_{card_type}_rois.png"
    cv2.imwrite(str(annotated_file), annotated)
    print("-" * 86)
    print(f"-> Ảnh tổng hợp đánh dấu ROI: {annotated_file}")
    print(f"-> Các ảnh crop chi tiết lưu tại: {crops_dir}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Debug và căn chỉnh ROI giấy tờ CCCD / GPLX")
    parser.add_argument("image_path", help="Đường dẫn đến file ảnh mẫu cần kiểm tra")
    parser.add_argument("--output-dir", default="debug_output", help="Thư mục xuất ảnh (mặc định: debug_output)")
    parser.add_argument("--card-type", choices=["cccd", "gplx"], default="cccd", help="Loại thẻ (cccd hoặc gplx)")
    parser.add_argument("--config", help="Đường dẫn file JSON chứa ROI tùy chỉnh (nếu có)")

    args = parser.parse_args()

    custom_rois = None
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                custom_rois = json.load(f)

    debug_rois(
        image_path=args.image_path,
        output_dir=args.output_dir,
        card_type=args.card_type,
        custom_rois=custom_rois,
    )


if __name__ == "__main__":
    main()

