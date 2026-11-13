"""Script gộp 2 dataset Roboflow (cccd.yolov8 và gplx.yolov8) thành dataset 2 lớp chuẩn YOLOv8.

Quy tắc thực hiện:
1. Giữ nguyên 100% dữ liệu gốc trong cccd.yolov8 và gplx.yolov8 (chỉ đọc, không sửa, không xóa).
2. Ánh xạ split: train -> train, valid -> val, test -> test.
3. Chuẩn hóa Class ID:
   - CCCD: Class 0 -> giữ nguyên 0 (tên: cccd).
   - GPLX: Class 0 -> chuyển thành 1 (tên: gplx).
4. Thêm tiền tố 'cccd_' và 'gplx_' vào tên file ảnh và nhãn TXT đích.
5. Giữ nguyên 4 tọa độ YOLO (x_center, y_center, width, height).
6. Hỗ trợ cờ --dry-run để chạy mô phỏng kiểm tra trước khi ghi file thật.
7. Cơ chế an toàn --overwrite chống ghi đè dữ liệu ngoài ý muốn.
8. Báo cáo thống kê toàn diện trước và sau khi gộp.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def compute_md5(file_path: Path) -> str:
    """Tính mã MD5 hash của file."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


class DatasetMerger:
    def __init__(
        self,
        cccd_dir: str | Path,
        gplx_dir: str | Path,
        output_dir: str | Path,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.cccd_dir = Path(cccd_dir).resolve()
        self.gplx_dir = Path(gplx_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.dry_run = dry_run
        self.overwrite = overwrite

        # Cấu hình nguồn: (thư mục nguồn, class_id_nguồn, class_id_đích, tiền tố)
        self.sources = [
            ("CCCD", self.cccd_dir, 0, 0, "cccd_"),
            ("GPLX", self.gplx_dir, 0, 1, "gplx_"),
        ]

    def validate_inputs(self) -> None:
        """Kiểm tra sự tồn tại và tính hợp lệ của các thư mục nguồn."""
        for name, src_path, _, _, _ in self.sources:
            if not src_path.exists():
                raise FileNotFoundError(f"Không tìm thấy thư mục dataset {name}: {src_path}")
            for roboflow_split in ["train", "valid", "test"]:
                img_dir = src_path / roboflow_split / "images"
                lbl_dir = src_path / roboflow_split / "labels"
                if not img_dir.exists() or not lbl_dir.exists():
                    raise FileNotFoundError(
                        f"Thiếu thư mục con {roboflow_split}/images hoặc {roboflow_split}/labels trong {src_path}"
                    )
        logger.info("Xác thực thư mục đầu vào thành công.")

    def run(self) -> dict:
        """Thực hiện quy trình kiểm tra và gộp dataset."""
        self.validate_inputs()

        logger.info(f"--- BẮT ĐẦU QUY TRÌNH GỘP DATASET (Dry-Run: {self.dry_run}) ---")
        logger.info(f"Nguồn CCCD: {self.cccd_dir}")
        logger.info(f"Nguồn GPLX: {self.gplx_dir}")
        logger.info(f"Đích:       {self.output_dir}\n")

        # Thống kê trước khi gộp
        source_stats = {}
        merged_plan = []  # Danh sách các thao tác copy / convert

        for doc_name, src_dir, src_cid, target_cid, prefix in self.sources:
            doc_stat = {"images": 0, "labels": 0, "boxes": 0, "splits": {}}
            for rf_split, out_split in SPLIT_MAP.items():
                img_dir = src_dir / rf_split / "images"
                lbl_dir = src_dir / rf_split / "labels"

                imgs = sorted([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
                lbls = sorted([f for f in lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"])

                img_map = {f.stem: f for f in imgs}
                lbl_map = {f.stem: f for f in lbls}

                common_stems = sorted(set(img_map.keys()).intersection(lbl_map.keys()))
                diff_img = set(img_map.keys()) - set(lbl_map.keys())
                diff_lbl = set(lbl_map.keys()) - set(img_map.keys())

                if diff_img:
                    logger.warning(f"[{doc_name} - {rf_split}] {len(diff_img)} ảnh thiếu nhãn: {list(diff_img)[:3]}")
                if diff_lbl:
                    logger.warning(f"[{doc_name} - {rf_split}] {len(diff_lbl)} nhãn thiếu ảnh: {list(diff_lbl)[:3]}")

                split_boxes = 0
                for stem in common_stems:
                    img_file = img_map[stem]
                    lbl_file = lbl_map[stem]

                    # Đọc và kiểm tra tọa độ label
                    lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                    new_lines = []
                    for line_no, line in enumerate(lines, 1):
                        tokens = line.strip().split()
                        if not tokens:
                            continue
                        if len(tokens) != 5:
                            raise ValueError(
                                f"Lỗi định dạng dòng {line_no} file {lbl_file}: {line}"
                            )
                        c_id = int(tokens[0])
                        xc, yc, w, h = map(float, tokens[1:])

                        # Xác thực tọa độ chuẩn hóa YOLO [0, 1]
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            raise ValueError(
                                f"Tọa độ không hợp lệ ở file {lbl_file} dòng {line_no}: {line}"
                            )

                        # Chuyển đổi class ID sang đích (CCCD -> 0, GPLX -> 1)
                        new_line = f"{target_cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                        new_lines.append(new_line)
                        split_boxes += 1

                    # Tên file đích có tiền tố cccd_ hoặc gplx_
                    target_stem = f"{prefix}{stem}"
                    target_img_name = f"{target_stem}{img_file.suffix}"
                    target_lbl_name = f"{target_stem}.txt"

                    dest_img_path = self.output_dir / "images" / out_split / target_img_name
                    dest_lbl_path = self.output_dir / "labels" / out_split / target_lbl_name

                    merged_plan.append({
                        "doc_type": doc_name,
                        "source_img": img_file,
                        "source_lbl": lbl_file,
                        "dest_img": dest_img_path,
                        "dest_lbl": dest_lbl_path,
                        "new_label_content": "\n".join(new_lines) + "\n",
                        "split": out_split,
                        "class_id": target_cid,
                        "box_count": len(new_lines),
                    })

                doc_stat["images"] += len(common_stems)
                doc_stat["labels"] += len(common_stems)
                doc_stat["boxes"] += split_boxes
                doc_stat["splits"][out_split] = {
                    "images": len(common_stems),
                    "boxes": split_boxes,
                }

            source_stats[doc_name] = doc_stat

        # Kiểm tra trùng lặp tên đích hoặc ghi đè
        dest_img_names = Counter([item["dest_img"].name for item in merged_plan])
        collisions = [name for name, count in dest_img_names.items() if count > 1]
        if collisions:
            raise RuntimeError(f"Xung đột tên file đích trùng lặp: {collisions[:5]}")

        # Kiểm tra file đích đã tồn tại trên đĩa (nếu chưa bật overwrite)
        existing_files = [
            item["dest_img"] for item in merged_plan if item["dest_img"].exists() and not self.overwrite
        ]
        if existing_files and not self.dry_run:
            raise FileExistsError(
                f"Phát hiện {len(existing_files)} file đích đã tồn tại (vd: {existing_files[0]}). "
                "Hãy thêm cờ --overwrite nếu muốn ghi đè."
            )

        # Kiểm tra rò rỉ dữ liệu chéo split (Data Leakage) bằng MD5 hash
        logger.info("Đang kiểm tra tính toàn vẹn và trùng lặp hash MD5 giữa các split...")
        hash_to_split = defaultdict(list)
        for item in merged_plan:
            h = compute_md5(item["source_img"])
            hash_to_split[h].append((item["dest_img"].name, item["split"]))

        leakage = []
        for h, entries in hash_to_split.items():
            splits_seen = set(e[1] for e in entries)
            if len(splits_seen) > 1:
                leakage.append((h, entries))

        if leakage:
            logger.warning(
                f"[CẢNH BÁO DATA LEAKAGE] Có {len(leakage)} nhóm ảnh trùng nội dung xuất hiện ở nhiều split khác nhau!"
            )
            for h, entries in leakage[:3]:
                logger.warning(f"  * Hash {h[:8]}: {entries}")
        else:
            logger.info("Tuyệt vời: Không có bất kỳ ảnh nào bị trùng lặp nội dung giữa các split.")

        # Thực thi ghi file nếu không phải dry-run
        copied_images = 0
        written_labels = 0
        total_boxes = 0
        class_summary = Counter()
        split_summary = defaultdict(lambda: {"images": 0, "labels": 0, "boxes": 0, "classes": Counter()})

        if not self.dry_run:
            logger.info("\nĐang tiến hành sao chép ảnh và ghi nhãn đã chuyển đổi Class ID...")
            for item in merged_plan:
                # Tạo thư mục cha nếu chưa có
                item["dest_img"].parent.mkdir(parents=True, exist_ok=True)
                item["dest_lbl"].parent.mkdir(parents=True, exist_ok=True)

                # Copy ảnh (bảo toàn nguyên vẹn ảnh gốc)
                shutil.copy2(item["source_img"], item["dest_img"])
                copied_images += 1

                # Ghi file nhãn mới với class ID đã chuẩn hóa
                item["dest_lbl"].write_text(item["new_label_content"], encoding="utf-8")
                written_labels += 1

                # Thống kê
                sp = item["split"]
                cid = item["class_id"]
                n_box = item["box_count"]
                total_boxes += n_box
                class_summary[cid] += n_box
                split_summary[sp]["images"] += 1
                split_summary[sp]["labels"] += 1
                split_summary[sp]["boxes"] += n_box
                split_summary[sp]["classes"][cid] += n_box
        else:
            logger.info("\n[DRY-RUN MODE] Mô phỏng thành công, không có file nào được ghi lên đĩa.")
            for item in merged_plan:
                copied_images += 1
                written_labels += 1
                sp = item["split"]
                cid = item["class_id"]
                n_box = item["box_count"]
                total_boxes += n_box
                class_summary[cid] += n_box
                split_summary[sp]["images"] += 1
                split_summary[sp]["labels"] += 1
                split_summary[sp]["boxes"] += n_box
                split_summary[sp]["classes"][cid] += n_box

        report = {
            "dry_run": self.dry_run,
            "source_stats": source_stats,
            "total_images": copied_images,
            "total_labels": written_labels,
            "total_boxes": total_boxes,
            "classes": dict(class_summary),
            "splits": {k: dict(v) for k, v in split_summary.items()},
            "data_leakage_count": len(leakage),
        }

        self.print_report(report)
        return report

    def print_report(self, report: dict) -> None:
        """In báo cáo thống kê trực quan."""
        print("\n" + "=" * 60)
        print(f"BÁO CÁO KẾT QUẢ GỘP DATASET (Dry-Run: {report['dry_run']})")
        print("=" * 60)
        print("1. Thống kê nguồn:")
        for doc_name, stat in report["source_stats"].items():
            print(f"   * {doc_name}: {stat['images']} ảnh, {stat['boxes']} boxes")
            for sp, sp_st in stat["splits"].items():
                print(f"       - {sp}: {sp_st['images']} ảnh ({sp_st['boxes']} boxes)")

        print("\n2. Thống kê dataset đích sau khi gộp:")
        print(f"   * Tổng số ảnh:       {report['total_images']}")
        print(f"   * Tổng số file nhãn: {report['total_labels']}")
        print(f"   * Tổng số BBox:      {report['total_boxes']}")
        print("   * Phân bố theo lớp:")
        print(f"       - Class 0 (cccd): {report['classes'].get(0, 0)} boxes")
        print(f"       - Class 1 (gplx): {report['classes'].get(1, 0)} boxes")

        print("\n3. Chi tiết theo từng split:")
        for sp in ["train", "val", "test"]:
            if sp in report["splits"]:
                sp_st = report["splits"][sp]
                c0 = sp_st["classes"].get(0, 0)
                c1 = sp_st["classes"].get(1, 0)
                print(f"   * [{sp.upper()}]: {sp_st['images']} ảnh | {sp_st['boxes']} boxes (CCCD: {c0}, GPLX: {c1})")

        print(f"\n4. Rò rỉ dữ liệu giữa các split: {report['data_leakage_count']} ảnh trùng.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Gộp dataset CCCD và GPLX cho YOLOv8n")
    parser.add_argument(
        "--cccd-dir",
        default=str(Path(__file__).resolve().parent.parent.parent.parent / "cccd.yolov8"),
        help="Đường dẫn thư mục cccd.yolov8",
    )
    parser.add_argument(
        "--gplx-dir",
        default=str(Path(__file__).resolve().parent.parent.parent.parent / "gplx.yolov8"),
        help="Đường dẫn thư mục gplx.yolov8",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "datasets"),
        help="Thư mục dataset đích",
    )
    parser.add_argument("--dry-run", action="store_true", help="Chạy mô phỏng kiểm tra, không ghi file")
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè nếu file đã tồn tại")

    args = parser.parse_args()

    merger = DatasetMerger(
        cccd_dir=args.cccd_dir,
        gplx_dir=args.gplx_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    merger.run()


if __name__ == "__main__":
    main()

