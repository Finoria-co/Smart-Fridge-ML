"""สรุปผลการเทรน 1 run → results/<out_name>/ (เข้า git ได้)

    python scripts/analyze_run.py runs/detect/baseline_n baseline_n

ทำ 3 อย่าง
1. คัดลอกไฟล์หลัก (results.csv, args.yaml, กราฟของ Ultralytics) ไป results/<out_name>/
2. กราฟ train loss เทียบ val loss (บท 03: ตรวจ overfitting) → loss_train_vs_val.png
3. พิมพ์สรุป: epoch ที่ดีที่สุด (ตาม fitness = 0.1·mAP50 + 0.9·mAP50-95 แบบเดียวกับที่ Ultralytics ใช้เลือก best.pt),
   ค่าที่ epoch นั้น, epoch ที่ val loss ต่ำสุด
"""
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["results.csv", "args.yaml", "results.png", "confusion_matrix_normalized.png",
        "BoxPR_curve.png", "BoxF1_curve.png"]
LOSSES = ["box_loss", "cls_loss", "dfl_loss"]


def main(run: Path, out_name: str):
    out = ROOT / "results" / out_name
    out.mkdir(parents=True, exist_ok=True)
    for f in KEEP:
        if (run / f).exists():
            shutil.copy2(run / f, out / f)

    df = pd.read_csv(run / "results.csv")
    df.columns = df.columns.str.strip()

    # 2) train vs val loss
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axes, LOSSES):
        ax.plot(df["epoch"], df[f"train/{name}"], label="train", color="#2a6fdb")
        ax.plot(df["epoch"], df[f"val/{name}"], label="val", color="#e07b00")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{out_name}: train vs val loss")
    fig.tight_layout()
    fig.savefig(out / "loss_train_vs_val.png", dpi=150)

    # 3) summary
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    b = df.loc[fitness.idxmax()]
    last = df.iloc[-1]
    print(f"epochs run: {len(df)} · total time: {last['time'] / 60:.1f} min")
    print(f"best epoch (fitness): {int(b['epoch'])}")
    for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        print(f"  {k:22s} best={b[k]:.4f}  last={last[k]:.4f}")
    for name in LOSSES:
        v = df[f"val/{name}"]
        print(f"  val/{name:9s} min at epoch {int(df.loc[v.idxmin(), 'epoch'])} ({v.min():.4f}) · last {v.iloc[-1]:.4f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
