"""หารูปใน test v2 ที่ predict_image() ทายวัตถุดิบ (Data Contract) ได้ตรงกับ label ทุกชนิด
เพื่อใช้เป็นตัวอย่าง demo ที่ "รันโมเดลจริง" แทน preset ที่เขียนผลไว้ล่วงหน้า

ไม่ได้ใช้ผลนี้ปรับโมเดลหรือ threshold (โมเดลและ conf 0.28 ถูกล็อกจาก val ไปแล้วใน Phase E)
เป็นการคัดรูปตัวอย่างสำหรับนำเสนอเท่านั้น ต้องบอกในสไลด์ว่าเป็นรูปที่คัดมา

    (WSL) python scripts/demo_pick.py [--data /root/data/foodseg_yolo_v2] [--split test]
ผล: results/demo_pick/candidates.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict import DEFAULT_CONF, aggregate, load_ingredient_map, load_model  # noqa: E402


def source_of(stem: str) -> str:
    for tag in ("aicook-lcv4d", "fridge-objects", "food-ingredients-x2", "meat-data-sets"):
        if tag in stem:
            return tag
    if stem.startswith("fridge_"):
        return "fridge_rf(v1)"
    if "thai-herbs" in stem:
        return "5-thai-herbs"
    return "foodseg103"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/data/foodseg_yolo_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    a = ap.parse_args()

    data = Path(a.data)
    names = yaml.safe_load(open(data / "data.yaml"))["names"]
    _, class_to_std = load_ingredient_map()
    imgs = sorted((data / "images" / a.split).iterdir())
    model = load_model()

    out = ROOT / "results" / "demo_pick"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(0, len(imgs), 32):
        batch = imgs[i:i + 32]
        for p, r in zip(batch, model.predict([str(x) for x in batch], conf=a.conf, verbose=False)):
            lbl = data / "labels" / a.split / (p.stem + ".txt")
            gt = {}
            for line in (lbl.read_text().splitlines() if lbl.exists() else []):
                if line.strip():
                    std = class_to_std.get(names[int(line.split()[0])])
                    if std:
                        gt[std] = gt.get(std, 0) + 1
            b = r.boxes
            dets = [(r.names[int(c)], s, bx) for c, s, bx in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
            pred = {g["ingredient"]: g for g in aggregate(dets)}
            rows.append({
                "file": p.name,
                "source": source_of(p.stem),
                "gt": ";".join(sorted(gt)),
                "pred": ";".join(sorted(pred)),
                "exact": int(set(gt) == set(pred) and len(gt) > 0),
                "n_gt": len(gt),
                "min_conf": round(min((g["confidence"] for g in pred.values()), default=0), 3),
                "pred_detail": ";".join(f"{k}x{g['count']}@{g['confidence']}" for k, g in sorted(pred.items())),
            })
    with open(out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    have = [r for r in rows if r["n_gt"] > 0]
    ex = [r for r in have if r["exact"]]
    print(f"{a.split}: {len(rows)} รูป · มีวัตถุดิบ Contract {len(have)} รูป · ทายชุดวัตถุดิบตรงทุกชนิด {len(ex)} รูป")
    print(f"  ในนั้นมี >= 2 ชนิด: {sum(r['n_gt'] >= 2 for r in ex)} รูป")


if __name__ == "__main__":
    main()
