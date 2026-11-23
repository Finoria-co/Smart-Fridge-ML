"""class ไหนถูกทายเป็นอะไรบ่อยที่สุด (จาก confusion matrix บน val) → results/<out_name>/confusions_val.csv

    python scripts/top_confusions.py runs/detect/<name>/weights/best.pt <name> [--classes pork "wild betel"]

แต่ละแถว: true class → predicted (หรือ background = ตรวจไม่เจอ) · จำนวน · สัดส่วนของ instance จริงของ class นั้น
แถว true = background คือ false positive (ไม่มีของจริง แต่โมเดลทายว่ามี)
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "/root/data/foodseg_yolo/data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("out_name")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--classes", nargs="*", help="พิมพ์เฉพาะ class เหล่านี้ (ไฟล์ csv มีทุก class)")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    from ultralytics import YOLO

    m = YOLO(args.weights).val(data=args.data, split="val", batch=16, workers=2, plots=True,  # plots=True จึงจะคำนวณ confusion matrix
                               verbose=False, project=str(ROOT / "runs" / "eval"),
                               name=f"{args.out_name}_val", exist_ok=True)
    cm = m.confusion_matrix.matrix  # [pred, true] ขนาด (nc+1, nc+1) แถว/คอลัมน์สุดท้าย = background
    names = dict(m.names)
    nc = len(names)
    label = lambda i: names[i] if i < nc else "background"

    rows = []
    for t in range(nc + 1):
        col = cm[:, t]
        total = col.sum()
        for pr in range(nc + 1):
            n = int(col[pr])
            if n == 0 or pr == t:
                continue
            rows.append({"true": label(t), "pred": label(pr), "count": n,
                         "share_of_true": round(n / total, 3) if t < nc and total else None})
    df = pd.DataFrame(rows).sort_values(["true", "count"], ascending=[True, False])
    out = ROOT / "results" / args.out_name
    df.to_csv(out / "confusions_val.csv", index=False)

    # false positive: ทาย class X ทั้งที่ไม่มีของจริง หรือเป็น class อื่น
    show = args.classes or []
    for c in show:
        cid = [k for k, v in names.items() if v == c][0]
        tp = int(cm[cid, cid])
        print(f"\n== {c}: ถูก {tp} จาก instance จริง {int(cm[:, cid].sum())}")
        print("  ของจริงคือ", c, "แต่ทายเป็น:")
        print(df[df.true == c].head(args.top).to_string(index=False))
        fp = [(label(t), int(cm[cid, t])) for t in range(nc + 1) if t != cid and cm[cid, t] > 0]
        fp.sort(key=lambda x: -x[1])
        print(f"  ทายว่าเป็น {c} แต่ของจริงคือ: {fp[:args.top]}")
    print(f"→ {out / 'confusions_val.csv'}")


if __name__ == "__main__":
    main()
