import json
from pathlib import Path

p = Path("debug_output/roi_calibration/roi_calibration_analysis.json")
data = json.loads(p.read_text(encoding="utf-8"))

for item in data:
    card_type = item["card_type"]
    fname = item["file"][:30]
    rot = item["rotation"]
    print(f"\n================ {card_type.upper()} ({fname}, rot={rot}°) ================")
    for t in item["tokens"]:
        txt = t["text"]
        b = t["norm_box"]
        conf = t["conf"]
        # Filter for key texts
        if any(k in txt.upper() for k in ["030", "300", "NGOC", "BAO", "2006", "NAM", "CHINH", "2031", "BICH", "GIAY", "CAN", "SO"]):
            print(f"  {txt:25} | box={b} | conf={conf}")

