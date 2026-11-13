"""Script kiểm tra chi tiết cấu trúc, tính toàn vẹn và thống kê của 2 dataset Roboflow:
- cccd.yolov8
- gplx.yolov8
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Tính difference hash (dHash) để phát hiện ảnh tương đồng / gần trùng."""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return 0


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def audit_single_dataset(name: str, root_dir: Path) -> dict:
    print(f"\n==================================================")
    print(f"BÁO CÁO KIỂM TRA DATASET: {name}")
    print(f"Thư mục: {root_dir.resolve()}")
    print(f"==================================================")

    splits = ["train", "valid", "test"]
    stats = {
        "name": name,
        "root": str(root_dir),
        "total_images": 0,
        "total_labels": 0,
        "splits": {},
        "duplicate_md5": [],
        "cross_split_filenames": [],
        "near_duplicates": [],
    }

    all_file_hashes = {}  # md5 -> list of (path, split)
    all_dhashes = {}      # path -> (dhash, split)
    stem_to_splits = defaultdict(list)

    for split in splits:
        img_dir = root_dir / split / "images"
        lbl_dir = root_dir / split / "labels"

        imgs = sorted([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]) if img_dir.exists() else []
        lbls = sorted([f for f in lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]) if lbl_dir.exists() else []

        stats["total_images"] += len(imgs)
        stats["total_labels"] += len(lbls)

        img_stems = {f.stem: f for f in imgs}
        lbl_stems = {f.stem: f for f in lbls}

        for stem in img_stems:
            stem_to_splits[stem].append(split)

        missing_lbl = [s for s in img_stems if s not in lbl_stems]
        missing_img = [s for s in lbl_stems if s not in img_stems]

        empty_lbl = []
        invalid_format = []
        invalid_coords = []
        class_counts = Counter()
        box_count = 0

        for stem, f in lbl_stems.items():
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                empty_lbl.append(f.name)
                continue
            lines = content.splitlines()
            for line_no, line in enumerate(lines, 1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) != 5:
                    invalid_format.append((f.name, line_no, line))
                    continue
                try:
                    cid = int(tokens[0])
                    xc, yc, w, h = map(float, tokens[1:])
                    class_counts[cid] += 1
                    box_count += 1
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        invalid_coords.append((f.name, line_no, (xc, yc, w, h)))
                except ValueError as e:
                    invalid_format.append((f.name, line_no, str(e)))

        # Tính MD5 và dHash
        for img_f in imgs:
            md5_val = hashlib.md5(img_f.read_bytes()).hexdigest()
            if md5_val not in all_file_hashes:
                all_file_hashes[md5_val] = []
            all_file_hashes[md5_val].append((img_f, split))
            dh = compute_dhash(img_f)
            all_dhashes[img_f] = (dh, split)

        stats["splits"][split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "boxes": box_count,
            "classes": dict(class_counts),
            "missing_labels": missing_lbl,
            "missing_images": missing_img,
            "empty_labels": empty_lbl,
            "invalid_format": invalid_format,
            "invalid_coords": invalid_coords,
        }

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  * Số lượng ảnh:  {len(imgs)}")
        print(f"  * Số file nhãn:  {len(lbls)}")
        print(f"  * Số Bounding Box: {box_count}")
        print(f"  * Phân bố Class:   {dict(class_counts)}")
        if missing_lbl:
            print(f"  [!] Ảnh thiếu nhãn: {len(missing_lbl)} file: {missing_lbl[:3]}")
        if missing_img:
            print(f"  [!] Nhãn thiếu ảnh: {len(missing_img)} file: {missing_img[:3]}")
        if empty_lbl:
            print(f"  [!] Nhãn rỗng:      {len(empty_lbl)} file: {empty_lbl[:3]}")
        if invalid_format:
            print(f"  [!] Sai định dạng:  {len(invalid_format)} dòng: {invalid_format[:3]}")
        if invalid_coords:
            print(f"  [!] Tọa độ ngoài [0,1]: {len(invalid_coords)} dòng: {invalid_coords[:3]}")

    # Kiểm tra ảnh trùng nội dung (MD5)
    dups = {h: flist for h, flist in all_file_hashes.items() if len(flist) > 1}
    stats["duplicate_md5"] = dups
    print(f"\n--- Kiểm tra trùng lặp nội dung (MD5) ---")
    print(f"  * Số nhóm ảnh trùng MD5 tuyệt đối: {len(dups)}")
    for h, flist in list(dups.items())[:5]:
        entries = [f"{p.name} ({sp})" for p, sp in flist]
        print(f"    - Hash {h[:8]}: {entries}")

    # Kiểm tra trùng tên giữa các split
    cross_splits = {stem: sps for stem, sps in stem_to_splits.items() if len(sps) > 1}
    stats["cross_split_filenames"] = cross_splits
    print(f"\n--- Kiểm tra trùng tên file giữa các split ---")
    print(f"  * Số tên file xuất hiện ở >1 split: {len(cross_splits)}")
    for stem, sps in list(cross_splits.items())[:5]:
        print(f"    - {stem}: {sps}")

    # Kiểm tra ảnh gần trùng (Perceptual similarity dHash giữa các split khác nhau)
    img_list = list(all_dhashes.items())
    near_dups = []
    # Chỉ kiểm tra giữa các split khác nhau (data leakage)
    for i in range(len(img_list)):
        p1, (dh1, sp1) = img_list[i]
        for j in range(i + 1, len(img_list)):
            p2, (dh2, sp2) = img_list[j]
            if sp1 != sp2 and dh1 != 0 and dh2 != 0:
                dist = hamming_distance(dh1, dh2)
                if dist <= 2:  # Gần trùng cao (hamming distance <= 2 trên 64-bit hash)
                    near_dups.append((p1.name, sp1, p2.name, sp2, dist))

    stats["near_duplicates"] = near_dups
    print(f"\n--- Khả năng rò rỉ dữ liệu (Data Leakage: ảnh gần trùng giữa các split) ---")
    print(f"  * Số cặp ảnh gần trùng giữa các split khác nhau: {len(near_dups)}")
    for p1, sp1, p2, sp2, dist in near_dups[:5]:
        print(f"    - {p1} ({sp1}) <--> {p2} ({sp2}) [Hamming dist = {dist}]")

    return stats


def main():
    root = Path(__file__).resolve().parent.parent.parent
    cccd_dir = root / "cccd.yolov8"
    gplx_dir = root / "gplx.yolov8"

    cccd_stats = audit_single_dataset("CCCD", cccd_dir)
    gplx_stats = audit_single_dataset("GPLX", gplx_dir)

    print("\n==================================================")
    print("TỔNG HỢP SO SÁNH HAI BỘ DỮ LIỆU GỐC")
    print("==================================================")
    print(f"- CCCD: Tổng {cccd_stats['total_images']} ảnh, {cccd_stats['total_labels']} nhãn.")
    print(f"- GPLX: Tổng {gplx_stats['total_images']} ảnh, {gplx_stats['total_labels']} nhãn.")


if __name__ == "__main__":
    main()

