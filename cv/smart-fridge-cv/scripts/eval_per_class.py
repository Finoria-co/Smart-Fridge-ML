"""mAP ราย class ของโมเดล บน split ที่เลือก (ค่าเริ่มต้น val) → results/<out_name>/per_class_<split>.csv

    python scripts/eval_per_class.py runs/detect/baseline_n/weights/best.pt baseline_n
    python scripts/eval_per_class.py <best.pt> <out_name> --split test   # Phase E เท่านั้น ใช้ครั้งเดียว

ตารางเรียงจาก mAP50-95 ต่ำ → สูง พร้อมจำนวนรูป/instance ใน split นั้นและใน train
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "/root/data/foodseg_yolo/data.yaml"


def train_counts(data_yaml: Path) -> dict[int, int]:
    """จำนวนรูปใน train ที่มีแต่ละ class"""
    counts: dict[int, int] = {}
    for f in (data_yaml.parent / "labels" / "train").glob("*.txt"):
        ids = {int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()}
        for i in ids:
            counts[i] = counts.get(i, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("out_name")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, batch=args.batch, workers=2,
                  plots=False, verbose=False, project=str(ROOT / "runs" / "eval"),
                  name=f"{args.out_name}_{args.split}", exist_ok=True)

    names = m.names
    tr = train_counts(Path(args.data))
    rows = []
    for i, c in enumerate(m.ap_class_index):
        p_, r_, ap50, ap = m.class_result(i)
        rows.append({"id": int(c), "class": names[int(c)],
                     "images": int(m.nt_per_image[c]), "instances": int(m.nt_per_class[c]),
                     "train_images": tr.get(int(c), 0),
                     "P": round(p_, 4), "R": round(r_, 4),
                     "mAP50": round(ap50, 4), "mAP50-95": round(ap, 4)})
    df = pd.DataFrame(rows).sort_values("mAP50-95")
    out = ROOT / "results" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"per_class_{args.split}.csv", index=False)

    print(f"all: P {m.box.mp:.4f} R {m.box.mr:.4f} mAP50 {m.box.map50:.4f} mAP50-95 {m.box.map:.4f}")
    print(df.to_string(index=False))
    print(f"→ {out / f'per_class_{args.split}.csv'}")


if __name__ == "__main__":
    main()
