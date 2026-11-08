"""Forwarder script tới scripts.predict_sample."""
from scripts.predict_sample import run_predict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dự đoán bounding box CCCD & GPLX")
    parser.add_argument("source", help="Đường dẫn file ảnh hoặc thư mục ảnh")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default="runs/detect/predict")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_predict(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=args.save_dir,
        show=args.show,
    )

