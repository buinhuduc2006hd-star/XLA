import json
from pathlib import Path

p = Path("debug_output/rectification/audit_stage3_geometry.json")
data = json.loads(p.read_text(encoding="utf-8"))

print(f"Total evaluated: {data['total']}")
print(f"YOLO detections: {data['detected_yolo']}")
print(f"CCCD: {data['cccd_count']}, GPLX: {data['gplx_count']}")
print(f"Rectify success: {data['rectify_success']}, Fallback: {data['rectify_fallback']}")
print(f"Methods: {data['methods']}")

large_rot = [d for d in data["details"] if abs(d.get("rotation_deg", 0)) > 20]
print(f"\nImages with |rotation_deg| > 20°: {len(large_rot)}")
for d in large_rot:
    print(f"  {d['file']}: rot={d['rotation_deg']}°, aspect={d['aspect_ratio']}, conf={d['rect_conf']}")

low_aspect = [d for d in data["details"] if d.get("aspect_ratio", 1.5) < 1.30]
print(f"\nImages with aspect_ratio < 1.30: {len(low_aspect)}")
for d in low_aspect:
    print(f"  {d['file']}: aspect={d['aspect_ratio']}, rot={d['rotation_deg']}°, conf={d['rect_conf']}")

low_conf = [d for d in data["details"] if d.get("rect_conf", 1.0) < 0.50]
print(f"\nImages with rect_conf < 0.50: {len(low_conf)}")
for d in low_conf:
    print(f"  {d['file']}: conf={d['rect_conf']}, method={d['rect_method']}")

