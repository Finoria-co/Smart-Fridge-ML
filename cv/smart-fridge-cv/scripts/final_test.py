"""Phase E: ประเมินโมเดลสุดท้าย (รันใน WSL)

1) เลือก confidence threshold จาก F1-confidence curve บน **val** (บท 08) -> results/final/conf_from_val.csv
   python scripts/final_test.py conf runs/detect/yolo11s_v2/weights/best.pt
2) วัดผลบน **test ครั้งเดียว** (ห้ามจูนอะไรหลังจากนี้) -> results/final/
   python scripts/final_test.py test runs/detect/yolo11s_v2/weights/best.pt
   - test v2 ทั้งชุด (1,427) + แยก 2 ส่วน: test เดิม v1 (1,317) และ test วัตถุดิบดิบที่เพิ่ม (110)
   - มีไฟล์ผลแล้วจะไม่รันซ้ำ (กันการดู test หลายรอบ)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
V1 = Path("/root/data/foodseg_yolo/data.yaml")
V2_DIR = Path("/root/data/foodseg_yolo_v2")
V2 = V2_DIR / "data.yaml"


def contract_classes():
    contract = json.loads((ROOT / "ingredient_map.json").read_text(encoding="utf-8"))["contract"]
    return {c: std for std, v in contract.items() for c in v["classes"]}


def per_class(m):
    rows = []
    for i, c in enumerate(m.ap_class_index):
        p_, r_, ap50, ap = m.class_result(i)
        rows.append({"id": int(c), "class": m.names[int(c)], "images": int(m.nt_per_image[c]),
                     "instances": int(m.nt_per_class[c]), "P": round(p_, 4), "R": round(r_, 4),
                     "mAP50": round(ap50, 4), "mAP50-95": round(ap, 4)})
    return pd.DataFrame(rows).sort_values("mAP50-95")


def run_val(model, data, split, name, plots):
    return model.val(data=str(data), split=split, batch=16, workers=2, plots=plots, verbose=False,
                     project=str(ROOT / "runs" / "eval"), name=name, exist_ok=True)


def stage_conf(model):
    """F1 = 2PR/(P+R) ต่อ class ที่ทุก threshold (Ultralytics คำนวณไว้ใน f1_curve) -> เลือก conf ที่ F1 เฉลี่ยสูงสุด"""
    m = run_val(model, V2, "val", "final_val_conf", plots=True)
    px = np.asarray(m.box.px)                 # threshold 0..1 (1000 จุด)
    f1 = np.asarray(m.box.f1_curve)           # (จำนวน class ที่มีใน val, 1000)
    names = [m.names[int(c)] for c in m.ap_class_index]
    cmap = contract_classes()
    idx_contract = [i for i, n in enumerate(names) if n in cmap]
    rows = []
    for label, idx in [("all classes", list(range(len(names)))), ("Data Contract classes", idx_contract)]:
        mean_f1 = f1[idx].mean(0)
        k = int(mean_f1.argmax())
        rows.append({"group": label, "n_classes": len(idx), "best_conf": round(float(px[k]), 3),
                     "mean_F1": round(float(mean_f1[k]), 4),
                     "mean_F1_at_0.25": round(float(mean_f1[np.abs(px - 0.25).argmin()]), 4)})
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "conf_from_val.csv", index=False)
    shutil.copy(Path(m.save_dir) / "BoxF1_curve.png", OUT / "val_F1_curve.png")
    print(df.to_string(index=False))


def stage_test(model, weights):
    if (OUT / "per_class_test.csv").exists():
        sys.exit(f"มี {OUT / 'per_class_test.csv'} แล้ว: test ใช้ได้ครั้งเดียว ไม่รันซ้ำ")
    # yaml ชี้ test เฉพาะรูปที่เพิ่ม (แบบเดียวกับ data_added_val.yaml)
    added = sorted(f"./images/test/{p.name}" for p in (V2_DIR / "images" / "test").glob("rfraw_*"))
    (V2_DIR / "added_test.txt").write_text("\n".join(added) + "\n")
    y = V2.read_text().replace("test: images/test\n", "test: added_test.txt\n")
    assert "added_test.txt" in y
    (V2_DIR / "data_added_test.yaml").write_text(y)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    cmap = contract_classes()
    for label, data, fname, plots in [("test v2 ทั้งชุด", V2, "per_class_test.csv", True),
                                      ("test เดิม v1", V1, "per_class_test_v1part.csv", False),
                                      ("test วัตถุดิบดิบที่เพิ่ม", V2_DIR / "data_added_test.yaml", "per_class_test_rawpart.csv", False)]:
        m = run_val(model, data, "test", f"final_{fname[:-4]}", plots)
        df = per_class(m)
        df.to_csv(OUT / fname, index=False)
        c = df[df["class"].isin(cmap)]
        n_img = (len(added) if "added" in str(data) else
                 sum(1 for _ in (Path(data).parent / "images" / "test").iterdir()))
        summary.append({"test": label, "images": n_img, "instances": int(m.nt_per_class.sum()),
                        "P": round(m.box.mp, 4), "R": round(m.box.mr, 4),
                        "mAP50": round(m.box.map50, 4), "mAP50-95": round(m.box.map, 4),
                        "contract_mean_mAP50-95": round(c["mAP50-95"].mean(), 4),
                        })
        if plots:
            for f in ["confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]:
                shutil.copy(Path(m.save_dir) / f, OUT / f"test_{f}")
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary_test.csv", index=False)
    (OUT / "README.md").write_text(f"weights: `{weights}` · รันครั้งเดียวด้วย `scripts/final_test.py test`\n", encoding="utf-8")
    print(s.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["conf", "test"])
    p.add_argument("weights")
    a = p.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    stage_conf(model) if a.stage == "conf" else stage_test(model, a.weights)


if __name__ == "__main__":
    main()
