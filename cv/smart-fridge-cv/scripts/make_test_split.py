"""
แบ่ง test set จาก val ของ dataset foodseg_fridge_thai_yolo (YOLO format, label แบบ polygon)

- ไม่แตะ train ที่เพื่อนคลีนมา: ย้ายรูปบางส่วนจาก val ไปเป็น test เท่านั้น
- แบ่งแบบ multi-label stratified (iterative stratification, Sechidis et al. 2011):
  รูปหนึ่งมีได้หลาย class จึงไล่จาก class ที่มีรูปน้อยที่สุดก่อน แล้วส่งรูปไปฝั่งที่ยัง "ขาด" class นั้นมากที่สุด
  ทำให้ทุก class ที่มีใน val ไปอยู่ทั้ง val และ test ตามสัดส่วนใกล้ TEST_FRAC มากที่สุด
- รูปที่ไม่มีวัตถุ (label ว่าง) แบ่งแบบสุ่มตามสัดส่วนเดียวกัน
- seed คงที่ รันซ้ำได้ผลเดิม · ถ้ามีโฟลเดอร์ test อยู่แล้วจะหยุด ไม่ทำซ้ำ

ใช้: python make_test_split.py <zip> <out_dir>
ผลลัพธ์: <out_dir>/foodseg_yolo/{images,labels}/{train,val,test} + data.yaml + split_report.md + test_files.txt
"""
import collections
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

TEST_FRAC = 0.40  # สัดส่วนของ val ที่ย้ายไปเป็น test (~10% ของ dataset ทั้งหมด)
SEED = 42


def safe_extract(zip_path, out_dir):
    """แตก zip โดยกันชื่อไฟล์ที่พยายามเขียนออกนอกโฟลเดอร์ปลายทาง"""
    out = Path(out_dir).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target = (out / m.filename).resolve()
            if out not in target.parents and target != out:
                raise ValueError(f"unsafe path in zip: {m.filename}")
        z.extractall(out)


def read_classes(label_file):
    text = Path(label_file).read_text(encoding="utf-8").strip()
    return {int(line.split()[0]) for line in text.splitlines() if line.strip()}


def iterative_stratify(img_classes, frac, rng):
    """img_classes: dict stem -> set(class) (ไม่รวมรูปว่าง) คืน set ของ stem ที่ไป test"""
    remaining = dict(img_classes)
    # ความต้องการของแต่ละฝั่ง ต่อ class และรวม
    counts = collections.Counter(c for cs in remaining.values() for c in cs)
    need = {"test": {c: n * frac for c, n in counts.items()}, "val": {c: n * (1 - frac) for c, n in counts.items()}}
    need_total = {"test": len(remaining) * frac, "val": len(remaining) * (1 - frac)}
    test = set()
    while remaining:
        left = collections.Counter(c for cs in remaining.values() for c in cs)
        c = min(left, key=lambda k: (left[k], k))  # class ที่เหลือรูปน้อยที่สุด
        stems = [s for s, cs in remaining.items() if c in cs]
        rng.shuffle(stems)
        for s in stems:
            # ไปฝั่งที่ยังต้องการ class นี้มากกว่า เสมอกันดูความต้องการรวม แล้วสุ่ม
            key = lambda side: (need[side][c], need_total[side], rng.random())
            side = max(("test", "val"), key=key)
            if side == "test":
                test.add(s)
            for k in remaining[s]:
                need[side][k] -= 1
            need_total[side] -= 1
            del remaining[s]
    return test


def main(zip_path, out_dir):
    out_dir = Path(out_dir)
    root = out_dir / "foodseg_yolo"
    if (root / "images" / "test").exists():
        sys.exit(f"มี {root / 'images' / 'test'} อยู่แล้ว ไม่แบ่งซ้ำ")
    if not root.exists():
        print("แตก zip ...")
        safe_extract(zip_path, out_dir)

    yaml_text = (root / "data.yaml").read_text(encoding="utf-8")
    names = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^\s+(\d+): (.+)$", yaml_text, re.M)}

    val_labels = sorted((root / "labels" / "val").glob("*.txt"))
    img_classes, empty = {}, []
    for lf in val_labels:
        cs = read_classes(lf)
        (img_classes.__setitem__(lf.stem, cs) if cs else empty.append(lf.stem))

    rng = np.random.default_rng(SEED)
    test = iterative_stratify(img_classes, TEST_FRAC, rng)
    empty = sorted(empty)
    rng.shuffle(empty)
    test |= set(empty[: round(len(empty) * TEST_FRAC)])

    # ย้ายรูป + label ของ test ออกจาก val
    for sub in ("images", "labels"):
        (root / sub / "test").mkdir(parents=True)
    for stem in sorted(test):
        img = next((root / "images" / "val").glob(stem + ".*"))
        shutil.move(str(img), root / "images" / "test" / img.name)
        shutil.move(str(root / "labels" / "val" / f"{stem}.txt"), root / "labels" / "test" / f"{stem}.txt")

    # data.yaml: ลบ path ออก -> Ultralytics ใช้โฟลเดอร์ที่ data.yaml อยู่ (path แบบ relative จะถูกตีความใต้ datasets_dir ของมัน)
    # ใช้ได้ทั้งในเครื่องและ Colab ไม่ว่าจะแตก dataset ไว้ที่ไหน + เพิ่ม test
    new_yaml = re.sub(r"^path: .*\n", "", yaml_text, flags=re.M)
    new_yaml = new_yaml.replace("val: images/val\n", "val: images/val\ntest: images/test\n")
    (root / "data.yaml").write_text(new_yaml, encoding="utf-8")
    (out_dir / "test_files.txt").write_text("\n".join(sorted(test)) + "\n", encoding="utf-8")

    # รายงาน: จำนวนรูปต่อ class ในแต่ละ split
    per = {}
    for split in ("train", "val", "test"):
        cnt, n, n_empty = collections.Counter(), 0, 0
        for lf in (root / "labels" / split).glob("*.txt"):
            cs = read_classes(lf)
            n += 1
            n_empty += not cs
            cnt.update(cs)
        per[split] = (cnt, n, n_empty)
    total = sum(v[1] for v in per.values())
    lines = [
        "# Split report: foodseg_fridge_thai_yolo",
        "",
        f"สร้างด้วย `scripts/make_test_split.py` · seed {SEED} · test = {TEST_FRAC:.0%} ของ val เดิม · train ไม่ถูกแตะ",
        "",
        "| split | รูป | % | รูปว่าง |",
        "|---|---|---|---|",
    ]
    for split, (_, n, ne) in per.items():
        lines.append(f"| {split} | {n:,} | {n / total:.1%} | {ne:,} |")
    lines += ["", "## รูปต่อ class (train / val / test) · ⚠️ = มีใน val เดิมแต่ val หรือ test เหลือ 0", "",
              "| id | class | train | val | test |", "|---|---|---|---|---|"]
    for c in sorted(names):
        tr, va, te = (per[s][0][c] for s in ("train", "val", "test"))
        flag = " ⚠️" if (va + te) > 0 and (va == 0 or te == 0) else ""
        lines.append(f"| {c} | {names[c]}{flag} | {tr} | {va} | {te} |")
    (out_dir / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"ย้าย {len(test)} รูปจาก val ไป test · รายงาน: {out_dir / 'split_report.md'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
