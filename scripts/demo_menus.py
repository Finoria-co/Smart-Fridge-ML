"""จัดกลุ่มรูปที่ YOLO ทายวัตถุดิบ Contract ตรงทุกชนิด (test + val) ตามเมนูอันดับ 1 ที่ได้
เมื่อส่ง "แบบที่เว็บรวมเรียกจริง" (ทุก class conf 0.25 · ชื่อไทยตาม Smart-Fridge-ML/modules/vision_detector.py)
เข้าตัวแนะนำเมนูของ fridge-chef-ai (จำลองด้วย numpy ใน demo_recommend.py)

    (WSL) python scripts/demo_menus.py <recipes_dataset.csv> <vision_detector.py>
ผล: results/demo_pick/menus.csv (1 แถวต่อรูป) + พิมพ์รูปที่ดีที่สุดต่อเมนู
"""
import ast
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_recommend import ROOT, STD_TO_SYSTEM_TH, load_recipes, recommend  # noqa: E402

FRIDGE = {"aicook-lcv4d", "fridge-objects", "fridge_rf(v1)"}


def main():
    recipes, vd = sys.argv[1], sys.argv[2]
    m = re.search(r"CLASS_MAP_ALL_EN_TO_TH = (\{.*?\n\})", open(vd, encoding="utf8").read(), re.S)
    all_th = ast.literal_eval(m.group(1))
    rows, X = load_recipes(recipes)
    out = []
    for split, fn in (("test", "candidates.csv"), ("val", "candidates_val.csv")):
        for c in csv.DictReader(open(ROOT / "results/demo_pick" / fn, encoding="utf8")):
            if c["exact"] != "1" or int(c["n_gt"]) < 2:
                continue
            th = [STD_TO_SYSTEM_TH.get(x) or all_th.get(x.replace("_", " ")) or x for x in c["web_all"].split(";")]
            b = recommend(th, rows, X, egg_guard=False)
            f = recommend(th, rows, X, egg_guard=True)
            out.append({
                "split": split, "file": c["file"], "source": c["source"], "fridge": int(c["source"] in FRIDGE),
                "contract_pred": c["pred"], "min_conf": c["min_conf"], "web_input_th": ",".join(th),
                "top1_fixed": f[0][1], "top1_fixed_pct": f[0][2], "top3_fixed": " / ".join(r[1] for r in f[:3]),
                "top1_buggy": b[0][1], "bug_changes_top1": int(b[0][1] != f[0][1]),
            })
    with open(ROOT / "results/demo_pick/menus.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"รูปที่ทาย Contract ตรงทุกชนิด (>=2 ชนิด): {len(out)} · ตู้เย็น {sum(o['fridge'] for o in out)}"
          f" · บั๊กไข่ไก่/คำย่อยเปลี่ยนเมนูอันดับ 1: {sum(o['bug_changes_top1'] for o in out)}")
    best = {}
    for o in sorted(out, key=lambda o: (-o["fridge"], -float(o["min_conf"]))):
        if o["bug_changes_top1"]:
            continue
        best.setdefault(o["top1_fixed"], []).append(o)
    for menu, lst in sorted(best.items(), key=lambda kv: -len(kv[1])):
        o = lst[0]
        print(f"{menu} ({len(lst)} รูป) · ดีสุด: [{o['split']}] {o['file'][:55]} · {o['source']} · "
              f"contract={o['contract_pred']} minconf={o['min_conf']}")


if __name__ == "__main__":
    main()
