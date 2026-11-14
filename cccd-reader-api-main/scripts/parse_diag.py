import json
from pathlib import Path

p = Path("debug_output/stage3_2/diagnosis_raw.json")
if not p.exists():
    p = Path("cccd-reader-api-main/debug_output/stage3_2/diagnosis_raw.json")

with open(p, encoding="utf-8") as f:
    data = json.load(f)

for k, items in data.items():
    print(f"\n================= {k} =================")
    for it in items:
        t = it["text"].lower()
        if any(w in t for w in ["030", "300", "553", "bảo", "bao", "27", "2006", "2031", "al", "a1", "khong", "thoi", "han", "hang", "nam", "nu", "nữ"]):
            print(f"  {it['bbox_norm']} (conf={it['conf']}): '{it['text']}'")

