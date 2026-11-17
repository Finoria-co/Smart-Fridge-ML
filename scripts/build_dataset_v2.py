"""สร้าง dataset v2 = v1 (foodseg_yolo) + รูปวัตถุดิบดิบจาก Roboflow

    python scripts/build_dataset_v2.py            # ทำทุกขั้น (ข้ามขั้นที่ทำแล้ว)
    python scripts/build_dataset_v2.py --dry-run  # แค่นับ ไม่ดาวน์โหลด SAM หรือเขียน v2

ขั้นตอน
1. ดาวน์โหลด version ที่กำหนดใน SOURCES → /root/data/rf_raw/<proj>_v<ver>/
2. map ชื่อ class ด้วย data/rf_class_map.yaml · รูปที่มี class = DROP ถูกตัดทั้งรูป
3. เลือกรูปที่มี class เป้าหมาย ≥1 (pork, shrimp, ham, bacon + เนื้อดิบอื่นที่สับสนกับ pork: steak, chicken duck)
4. ตัดรูปที่ซ้ำกับ v1 (เทียบชื่อไฟล์ต้นฉบับ) · 1 รูปต่อ 1 ต้นฉบับ (Roboflow augment เป็น <ต้นฉบับ>.rf.<hash>)
5. จำกัดไม่เกิน CAP รูปต่อ class เป้าหมาย
6. กล่อง → polygon ด้วย SAM (ultralytics yolo_bbox2segment) ให้ format เท่า v1
7. แบ่ง train/val/test 70/15/15 ด้วย iterative_stratify (seed 42) จาก make_test_split.py
8. รวมกับ v1 (hardlink) → /root/data/foodseg_yolo_v2/ + data.yaml, data_added_val.yaml (val เฉพาะรูปที่เพิ่ม)
9. รายงาน data/split_report_v2.md, data/v2_added_files.txt, runs/rf_preview/v2_added_*.jpg
"""
import argparse
import collections
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_test_split import iterative_stratify  # noqa: E402
from rf_inspect import api_key  # noqa: E402

V1 = Path("/root/data/foodseg_yolo")
V2 = Path("/root/data/foodseg_yolo_v2")
RAW = Path("/root/data/rf_raw")
STAGE = Path("/root/data/rf_stage")
SEED = 42
CAP = 300
SOURCES = [  # (workspace, project, version, ชื่อสั้นใน class map)
    ("karel-cornelis-q2qqg", "aicook-lcv4d", 1, "aicook-lcv4d"),
    ("fooddetection-essdj", "fridge-objects", 9, "fridge-objects"),
    ("yolov5-rznzw", "food-ingredients-x2", 1, "food-ingredients-x2"),
    ("thesis-6tgwg", "meat-data-sets", 4, "meat-data-sets"),
]
PRIMARY = ["pork", "shrimp", "ham", "bacon"]  # ที่ต้องการเพิ่มจริง
RELATED = ["steak", "chicken duck"]  # เนื้อดิบที่สับสนกับ pork (กันโมเดลเห็นเนื้อดิบ = pork อย่างเดียว)


def orig_stem(name: str) -> str:
    """ชื่อรูปต้นฉบับ: ตัด prefix ของ v1 และ .rf.<hash> ของ Roboflow"""
    s = re.sub(r"^(fridge|foodseg|5-thai-herbs)_(train|valid|test)_", "", Path(name).stem)
    return re.sub(r"\.rf\.[0-9a-f]+(_r\d+)?$", "", s).lower()


def names_of(yaml_path: Path) -> list[str]:
    n = yaml.safe_load(yaml_path.read_text())["names"]
    return list(n.values()) if isinstance(n, dict) else list(n)


def download(key: str):
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    for ws, proj, ver, _ in SOURCES:
        loc = RAW / f"{proj}_v{ver}"
        if (loc / "data.yaml").exists():
            print(f"มีแล้ว: {loc}")
            continue
        print(f"ดาวน์โหลด {ws}/{proj} v{ver} ...", flush=True)
        rf.workspace(ws).project(proj).version(ver).download("yolov11", location=str(loc), overwrite=True)


def collect(ours: dict[str, int]):
    """คืน list ของรูปที่เลือก: (source, img_path, [(cls_id, coords...)]) + สถิติ"""
    cmap = yaml.safe_load((ROOT / "data" / "rf_class_map.yaml").read_text())
    v1_stems = {orig_stem(p.name) for p in V1.glob("images/*/*")}
    want = {ours[c] for c in PRIMARY + RELATED}
    stats = collections.Counter()
    picked, seen = [], set()
    for _, proj, ver, short in SOURCES:
        src = RAW / f"{proj}_v{ver}"
        rf_names = names_of(src / "data.yaml")
        for lf in sorted(src.glob("*/labels/*.txt")):
            stem = orig_stem(lf.name)
            key = (short, stem)
            stats[f"{short}: ไฟล์ทั้งหมด"] += 1
            if key in seen:
                stats[f"{short}: ซ้ำต้นฉบับเดิม (augment)"] += 1
                continue
            seen.add(key)
            if stem in v1_stems:
                stats[f"{short}: ซ้ำกับ v1"] += 1
                continue
            rows, drop = [], False
            for line in lf.read_text().splitlines():
                v = line.split()
                if not v:
                    continue
                new = cmap[short][rf_names[int(v[0])]]
                if new == "DROP":
                    drop = True
                    break
                rows.append([ours[new]] + v[1:])
            if drop:
                stats[f"{short}: ตัดเพราะมี class DROP"] += 1
                continue
            if not want & {r[0] for r in rows}:
                stats[f"{short}: ไม่มี class เป้าหมาย"] += 1
                continue
            img = next(lf.parent.parent.joinpath("images").glob(lf.stem + ".*"))
            picked.append((short, img, rows))
            stats[f"{short}: เลือก"] += 1
    return picked, stats


def cap(picked, ours, rng):
    """จำกัดจำนวนรูปต่อ class เป้าหมาย ไล่จาก class ที่มีน้อยก่อน (รูปที่มีหลายเป้าหมายนับทุกตัว)"""
    tid = [ours[c] for c in PRIMARY + RELATED]
    order = list(range(len(picked)))
    rng.shuffle(order)
    keep, count = [], collections.Counter()
    for i in order:
        cls = {r[0] for r in picked[i][2]} & set(tid)
        if any(count[c] < CAP for c in cls):
            keep.append(i)
            count.update(cls)
    return [picked[i] for i in sorted(keep)]


def stage(picked):
    """เขียนรูป + label กล่องลง STAGE แล้วแปลงเป็น polygon ด้วย SAM"""
    from ultralytics.data.converter import yolo_bbox2segment

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "images").mkdir(parents=True)
    (STAGE / "labels").mkdir()
    names = {}
    for short, img, rows in picked:
        new = f"rfraw_{short}_{orig_stem(img.name)}{img.suffix.lower()}"
        new = re.sub(r"[^\w.\-]", "_", new)
        shutil.copy2(img, STAGE / "images" / new)
        (STAGE / "labels" / (Path(new).stem + ".txt")).write_text(
            "\n".join(" ".join(map(str, r)) for r in rows) + "\n")
        names[new] = short
    boxes = sum(1 for f in (STAGE / "labels").glob("*.txt")
                for l in f.read_text().splitlines() if len(l.split()) == 5)
    print(f"stage: {len(names)} รูป · กล่องที่ต้องแปลง {boxes}", flush=True)
    yolo_bbox2segment(im_dir=str(STAGE / "images"), save_dir=str(STAGE / "labels_seg"), sam_model="sam_b.pt")
    # SAM บางกล่องอาจได้ polygon ว่าง → ใช้สี่เหลี่ยมของกล่องเดิมแทน (ไม่ให้ label หาย)
    fixed = 0
    for lf in (STAGE / "labels").glob("*.txt"):
        seg = STAGE / "labels_seg" / lf.name
        box_rows = [l.split() for l in lf.read_text().splitlines() if l.split()]
        seg_rows = [l.split() for l in seg.read_text().splitlines() if l.split()] if seg.exists() else []
        if len(seg_rows) != len(box_rows):
            seg_rows = []
            for r in box_rows:
                c, x, y, w, h = r[0], *map(float, r[1:5])
                seg_rows.append([c] + [f"{v:.6f}" for v in (x - w / 2, y - h / 2, x + w / 2, y - h / 2,
                                                           x + w / 2, y + h / 2, x - w / 2, y + h / 2)])
            fixed += 1
        seg.write_text("\n".join(" ".join(r) for r in seg_rows) + "\n")
    print(f"SAM เสร็จ · รูปที่ใช้สี่เหลี่ยมแทนเพราะจำนวน polygon ไม่ตรง: {fixed}")
    return names


def split(stems_cls: dict[str, set[int]], rng):
    test = iterative_stratify(stems_cls, 0.15, rng)
    rest = {s: c for s, c in stems_cls.items() if s not in test}
    val = iterative_stratify(rest, 0.15 / 0.85, rng)
    return {s: ("test" if s in test else "val" if s in val else "train") for s in stems_cls}


def merge(assign: dict[str, str]):
    if V2.exists():
        shutil.rmtree(V2)
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (V2 / sub / sp).mkdir(parents=True)
            for f in (V1 / sub / sp).iterdir():
                os.link(f, V2 / sub / sp / f.name)  # hardlink: ไม่กินพื้นที่เพิ่ม
    for img in (STAGE / "images").iterdir():
        sp = assign[img.stem]
        shutil.copy2(img, V2 / "images" / sp / img.name)
        shutil.copy2(STAGE / "labels_seg" / f"{img.stem}.txt", V2 / "labels" / sp / f"{img.stem}.txt")
    shutil.copy2(V1 / "data.yaml", V2 / "data.yaml")
    added_val = sorted(f"./images/val/{p.name}" for p in (V2 / "images" / "val").glob("rfraw_*"))
    (V2 / "added_val.txt").write_text("\n".join(added_val) + "\n")
    y = (V1 / "data.yaml").read_text().replace("val: images/val\n", "val: added_val.txt\n")
    (V2 / "data_added_val.yaml").write_text(y)


def report(picked, assign, stats, src_of, ours_names):
    def counts(split_dir):
        c = collections.Counter()
        for lf in split_dir.glob("*.txt"):
            c.update({int(l.split()[0]) for l in lf.read_text().splitlines() if l.split()})
        return c

    v1c = {sp: counts(V1 / "labels" / sp) for sp in ("train", "val", "test")}
    add = {sp: collections.Counter() for sp in ("train", "val", "test")}
    for stem, sp in assign.items():
        lf = STAGE / "labels_seg" / f"{stem}.txt"
        add[sp].update({int(l.split()[0]) for l in lf.read_text().splitlines() if l.split()})
    n_add = collections.Counter(assign.values())
    lines = ["# Split report: dataset v2 (v1 + วัตถุดิบดิบจาก Roboflow)", "",
             f"สร้างด้วย `scripts/build_dataset_v2.py` · seed {SEED} · รูปใหม่แบ่ง 70/15/15 · v1 ไม่เปลี่ยน · CAP {CAP} รูป/class เป้าหมาย",
             "ที่มา: `data/sources.md` · map class: `data/rf_class_map.yaml`", "",
             "| split | v1 | เพิ่ม | v2 |", "|---|---|---|---|"]
    for sp in ("train", "val", "test"):
        a = len(list((V1 / "images" / sp).iterdir()))
        lines.append(f"| {sp} | {a:,} | {n_add[sp]:,} | {a + n_add[sp]:,} |")
    lines += ["", "## การคัดรูปจาก Roboflow", ""] + [f"- {k}: {v:,}" for k, v in sorted(stats.items())]
    lines += ["", "## รูปต่อ class ที่เปลี่ยน (v1 → เพิ่ม) แยก train / val / test", "",
              "| id | class | train v1 | +train | val v1 | +val | test v1 | +test |", "|---|---|---|---|---|---|---|---|"]
    for cid in sorted(ours_names):
        if not any(add[sp][cid] for sp in add):
            continue
        r = [v1c[sp][cid] for sp in ("train", "val", "test")]
        a = [add[sp][cid] for sp in ("train", "val", "test")]
        lines.append(f"| {cid} | {ours_names[cid]} | {r[0]} | +{a[0]} | {r[1]} | +{a[1]} | {r[2]} | +{a[2]} |")
    (ROOT / "data" / "split_report_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "data" / "v2_added_files.txt").write_text(
        "\n".join(f"{assign[Path(n).stem]}\t{src_of[n]}\t{n}" for n in sorted(src_of)) + "\n", encoding="utf-8")
    print("\n".join(lines))


def grids(assign, ours_names, per=12):
    """กริด polygon ของรูปที่เพิ่ม แยกตามที่มา → runs/rf_preview/v2_added_<source>.jpg"""
    import cv2

    by_src = collections.defaultdict(list)
    for img in sorted((STAGE / "images").iterdir()):
        by_src[img.name.split("_")[1]].append(img)
    rng = np.random.default_rng(0)
    out_dir = ROOT / "runs" / "rf_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, imgs in by_src.items():
        tiles = []
        for img in rng.choice(imgs, min(per, len(imgs)), replace=False):
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            for l in (STAGE / "labels_seg" / f"{img.stem}.txt").read_text().splitlines():
                v = l.split()
                pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.polylines(im, [pts], True, (0, 0, 255), 3)
                cv2.putText(im, ours_names[int(v[0])], tuple(pts[0]), 0, 0.9, (0, 255, 255), 2)
            tiles.append(cv2.resize(im, (400, 400)))
        while len(tiles) % 4:
            tiles.append(np.zeros((400, 400, 3), np.uint8))
        cv2.imwrite(str(out_dir / f"v2_added_{src}.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(f"กริด → {out_dir}/v2_added_*.jpg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ours_names = dict(enumerate(names_of(V1 / "data.yaml")))
    ours = {v: k for k, v in ours_names.items()}

    download(api_key())
    picked, stats = collect(ours)
    rng = np.random.default_rng(SEED)
    picked = cap(picked, ours, rng)
    stats["รวมหลัง CAP"] = len(picked)
    tc = collections.Counter(c for _, _, rows in picked for c in {r[0] for r in rows})
    print({ours_names[ours[c]]: tc[ours[c]] for c in PRIMARY + RELATED}, dict(stats))
    if args.dry_run:
        return

    src_of = stage(picked)
    stems_cls = {}
    for lf in (STAGE / "labels_seg").glob("*.txt"):
        stems_cls[lf.stem] = {int(l.split()[0]) for l in lf.read_text().splitlines() if l.split()}
    assign = split(stems_cls, np.random.default_rng(SEED))
    merge(assign)
    report(picked, assign, stats, src_of, ours_names)
    grids(assign, ours_names)


if __name__ == "__main__":
    main()
