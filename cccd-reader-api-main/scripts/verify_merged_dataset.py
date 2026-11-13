"""Script xác minh toàn diện dataset sau khi gộp:
1. So khớp số lượng nguồn vs đích
2. Kiểm tra Class ID (Class 0: CCCD, Class 1: GPLX)
3. Kiểm tra tính hợp lệ của từng Bounding Box
4. Kiểm tra trực quan ngẫu nhiên (vẽ bbox và lưu ảnh mẫu)
5. Kiểm tra nạp dữ liệu bằng YOLO DataLoader thực tế
"""

import random
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
CONFIG_FILE = PROJECT_ROOT / "configs" / "data.yaml"
OUTPUT_DEBUG_DIR = PROJECT_ROOT / "debug_output" / "merged_samples"


def verify_dataset():
    print("==================================================")
    print("BẮT ĐẦU KIỂM TRA TOÀN DIỆN DATASET SAU KHI GỘP")
    print(f"Dataset root: {DATASETS_DIR}")
    print(f"Config path:  {CONFIG_FILE}")
    print("==================================================\n")

    splits = ["train", "val", "test"]
    expected_counts = {
        "train": {"cccd": 98, "gplx": 218, "total": 316},
        "val": {"cccd": 28, "gplx": 62, "total": 90},
        "test": {"cccd": 14, "gplx": 32, "total": 46},
    }

    total_images = 0
    total_labels = 0
    total_boxes = 0
    class_counter = Counter()
    errors = []

    # 1. Kiểm tra từng split
    for split in splits:
        img_dir = DATASETS_DIR / "images" / split
        lbl_dir = DATASETS_DIR / "labels" / split

        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        total_images += len(imgs)
        total_labels += len(lbls)

        img_map = {f.stem: f for f in imgs}
        lbl_map = {f.stem: f for f in lbls}

        # Check 1:1 match
        if set(img_map.keys()) != set(lbl_map.keys()):
            diff_img = set(img_map.keys()) - set(lbl_map.keys())
            diff_lbl = set(lbl_map.keys()) - set(img_map.keys())
            errors.append(f"[{split}] Lệch tên file: {len(diff_img)} ảnh thiếu nhãn, {len(diff_lbl)} nhãn thiếu ảnh")

        # Check counts
        cccd_imgs = [s for s in img_map if s.startswith("cccd_")]
        gplx_imgs = [s for s in img_map if s.startswith("gplx_")]

        exp = expected_counts[split]
        if len(cccd_imgs) != exp["cccd"] or len(gplx_imgs) != exp["gplx"]:
            errors.append(
                f"[{split}] Số lượng không khớp mong đợi! Nhận được: CCCD={len(cccd_imgs)}, GPLX={len(gplx_imgs)}. "
                f"Mong đợi: CCCD={exp['cccd']}, GPLX={exp['gplx']}"
            )

        # Check label content & class IDs
        split_boxes = 0
        for stem, lbl_path in lbl_map.items():
            is_cccd = stem.startswith("cccd_")
            is_gplx = stem.startswith("gplx_")

            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                errors.append(f"[{split}] File nhãn rỗng: {lbl_path.name}")
                continue

            for line_no, line in enumerate(lines, 1):
                tokens = line.strip().split()
                if len(tokens) != 5:
                    errors.append(f"[{split}] {lbl_path.name}:{line_no} sai số trường: {line}")
                    continue
                cid = int(tokens[0])
                xc, yc, w, h = map(float, tokens[1:])

                # Verify class mapping
                if is_cccd and cid != 0:
                    errors.append(f"[{split}] File CCCD {lbl_path.name} chứa class ID sai: {cid} (cần 0)")
                if is_gplx and cid != 1:
                    errors.append(f"[{split}] File GPLX {lbl_path.name} chứa class ID sai: {cid} (cần 1)")

                # Verify coordinates
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"[{split}] {lbl_path.name}:{line_no} tọa độ ngoài [0, 1]: ({xc}, {yc}, {w}, {h})")

                class_counter[cid] += 1
                split_boxes += 1
                total_boxes += 1

        print(f"Split [{split.upper()}]:")
        print(f"  - Ảnh:  {len(imgs)} (CCCD: {len(cccd_imgs)}, GPLX: {len(gplx_imgs)})")
        print(f"  - Nhãn: {len(lbls)}")
        print(f"  - BBox: {split_boxes}\n")

    print("--------------------------------------------------")
    print("TỔNG KẾT SAU KHI XÁC MINH:")
    print(f"- Tổng số ảnh:       {total_images} (Kỳ vọng: 452)")
    print(f"- Tổng số nhãn:      {total_labels} (Kỳ vọng: 452)")
    print(f"- Tổng số Bounding Box: {total_boxes} (Kỳ vọng: 452)")
    print(f"- Class 0 (cccd):    {class_counter[0]} (Kỳ vọng: 140)")
    print(f"- Class 1 (gplx):    {class_counter[1]} (Kỳ vọng: 312)")
    print(f"- Số lỗi phát hiện:  {len(errors)}")

    if errors:
        print("\nDANH SÁCH LỖI:")
        for e in errors[:10]:
            print(f"  [X] {e}")
        raise RuntimeError("Xác minh thất bại do có lỗi dữ liệu!")
    else:
        print(">> TẤT CẢ CÁC ĐIỀU KIỆN DỮ LIỆU ĐÃ ĐẠT 100%!")

    # 2. Kiểm tra trực quan ngẫu nhiên (Visual Check)
    print("\n--------------------------------------------------")
    print("TIẾN HÀNH RENDER ẢNH MẪU TRỰC QUAN...")
    OUTPUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    colors = {0: (46, 139, 87), 1: (218, 112, 37)}  # Green for CCCD, Orange/Blue for GPLX
    names = {0: "CCCD", 1: "GPLX"}

    sample_count = 0
    for split in splits:
        img_dir = DATASETS_DIR / "images" / split
        lbl_dir = DATASETS_DIR / "labels" / split

        for prefix, cid in [("cccd_", 0), ("gplx_", 1)]:
            candidates = sorted(list(img_dir.glob(f"{prefix}*.*")))
            if candidates:
                samples = random.sample(candidates, min(2, len(candidates)))
                for s_img in samples:
                    s_lbl = lbl_dir / f"{s_img.stem}.txt"
                    img = cv2.imread(str(s_img))
                    ih, iw = img.shape[:2]

                    for line in s_lbl.read_text().splitlines():
                        c, xc, yc, w, h = line.split()
                        c = int(c)
                        xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                        x1 = int((xc - w / 2) * iw)
                        y1 = int((yc - h / 2) * ih)
                        x2 = int((xc + w / 2) * iw)
                        y2 = int((yc + h / 2) * ih)

                        cv2.rectangle(img, (x1, y1), (x2, y2), colors[c], 3)
                        cv2.putText(
                            img,
                            f"{names[c]} (Class {c})",
                            (x1 + 5, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            colors[c],
                            2,
                        )

                    out_path = OUTPUT_DEBUG_DIR / f"vis_{split}_{s_img.name}"
                    cv2.imwrite(str(out_path), img)
                    sample_count += 1

    print(f"Đã xuất {sample_count} ảnh trực quan mẫu vào: {OUTPUT_DEBUG_DIR}")

    # 3. Kiểm tra YOLO DataLoader
    print("\n--------------------------------------------------")
    print("KIỂM TRA YOLO DATALOADER TRỰC TIẾP...")
    import os
    os.chdir(PROJECT_ROOT)
    data_dict = check_det_dataset("configs/data.yaml")
    print("YOLO check_det_dataset parsed:")
    print(f"  * train: {data_dict['train']}")
    print(f"  * val:   {data_dict['val']}")
    print(f"  * test:  {data_dict['test']}")
    print(f"  * nc:    {data_dict['nc']}")
    print(f"  * names: {data_dict['names']}")

    # Khởi tạo model và kiểm tra dataloader val 1 batch
    model = YOLO("yolov8n.pt")
    print("\nKhởi tạo YOLO validation test batch (dry-run dataloader)...")
    val_results = model.val(
        data="configs/data.yaml",
        split="val",
        batch=4,
        imgsz=320,  # test nhẹ
        device="cpu",
        max_det=5,
        plots=False,
    )
    print("YOLO DataLoader nạp thành công 100%! val metrics:", val_results.box.map50)
    print("==================================================")


if __name__ == "__main__":
    verify_dataset()

