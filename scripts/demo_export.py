"""คัดลอกรูปทั้งหมดใน results/demo_pick/menus.csv (ทาย Contract ตรงทุกชนิด >= 2 ชนิด) ไปไว้ทดสอบระบบ
ชื่อไฟล์ใหม่ขึ้นต้นด้วยเลขลำดับ (ห้ามขึ้นต้น fridge_ เพราะเว็บรวมจะข้ามโมเดลไปใช้ preset)
    (WSL) python scripts/demo_export.py
ผล: demo_test_images/ + manifest.csv (ไม่เข้า git: รูปเป็นของ dataset ภายนอก)
"""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/root/data/foodseg_yolo_v2/images")
out = ROOT / "demo_test_images"
shutil.rmtree(out, ignore_errors=True)
out.mkdir()
rows = list(csv.DictReader(open(ROOT / "results/demo_pick/menus.csv", encoding="utf8")))
rows.sort(key=lambda r: (-int(r["fridge"]), r["top1_fixed"], -float(r["min_conf"])))
man = []
for i, r in enumerate(rows, 1):
    src = DATA / r["split"] / r["file"]
    kind = "fridge" if r["fridge"] == "1" else ("dish" if r["source"] == "foodseg103" else "raw")
    new = f"{i:03d}_{kind}_{src.stem[:40]}{src.suffix}"
    shutil.copy(src, out / new)
    man.append({"new_file": new, **r})
with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(man[0]))
    w.writeheader()
    w.writerows(man)
print(len(man), "รูป ->", out)
