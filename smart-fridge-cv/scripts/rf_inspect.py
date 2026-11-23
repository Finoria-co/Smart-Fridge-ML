"""สำรวจ dataset ผู้สมัครบน Roboflow Universe: จำนวนรูป, class + จำนวน, version, license

    python scripts/rf_inspect.py                       # ผู้สมัครในรายการ CANDIDATES
    python scripts/rf_inspect.py ws/project ws2/proj2  # ระบุเอง
    python scripts/rf_inspect.py --grid ws/project     # วาดกริดรูปตัวอย่าง 12 รูป → runs/rf_preview/

API key: env ROBOFLOW_API_KEY หรือไฟล์ C:\\Users\\patim\\.roboflow_key (WSL: /mnt/c/Users/patim/.roboflow_key)
ห้ามใส่ key ในไฟล์ของ repo
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_FILES = [Path("/mnt/c/Users/patim/.roboflow_key"), Path.home() / ".roboflow_key"]
TARGET = re.compile(r"pork|shrimp|prawn|ham\b|bacon|meat|beef|chicken|sausage", re.I)

CANDIDATES = [
    "recipe-raw/object-detection-raw-ingredient",
    "data-science-lab-final-project/recipe-generator-sqz2j",
    "project-qmmac/raw-ingredieant",
    "objectdetectionlearning-4empo/aicook-lcv4d-zzbgs",
    "yolov5-rznzw/food-ingredients-x2",
    "thesis-6tgwg/meat-data-sets",
    "shrimpfs/shrimp-ivurm",
]


def api_key() -> str:
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"].strip()
    for f in KEY_FILES:
        if f.exists():
            return f.read_text(encoding="utf-8-sig").strip()
    raise SystemExit("ไม่พบ API key: ตั้ง ROBOFLOW_API_KEY หรือสร้างไฟล์ C:\\Users\\patim\\.roboflow_key")


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def inspect(slug: str, key: str) -> dict:
    try:
        d = get(f"https://api.roboflow.com/{slug}?api_key={key}")
    except urllib.error.HTTPError as e:
        return {"slug": slug, "error": f"HTTP {e.code}"}
    p = d.get("project", {})
    versions = d.get("versions", [])
    classes = p.get("classes", {}) or {}
    return {
        "slug": slug,
        "name": p.get("name"),
        "type": p.get("type"),
        "images": p.get("images"),
        "license": p.get("license"),
        "versions": [v.get("id", "").split("/")[-1] for v in versions],
        "classes": classes,
        "targets": {k: v for k, v in classes.items() if TARGET.search(k)},
    }


def grid(slug: str, key: str, n: int = 12):
    """ดาวน์โหลด version ล่าสุด (yolov11) แล้ววาดกล่อง/polygon บนรูปตัวอย่าง n รูปที่มี target"""
    import random

    import cv2
    import numpy as np
    import yaml
    from roboflow import Roboflow

    ws, proj = slug.split("/")
    project = Roboflow(api_key=key).workspace(ws).project(proj)
    ver = max(int(v.version) for v in project.versions())
    loc = Path("/root/data/rf_raw") / proj
    if not (loc / "data.yaml").exists():
        project.version(ver).download("yolov11", location=str(loc), overwrite=True)
    names = yaml.safe_load((loc / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    tgt = {i for i, nm in enumerate(names) if TARGET.search(nm)}
    labels = [lf for lf in loc.glob("*/labels/*.txt")
              if any(int(l.split()[0]) in tgt for l in lf.read_text().splitlines() if l.split())]
    random.seed(0)
    tiles = []
    for lf in random.sample(labels, min(n, len(labels))):
        img_p = next(lf.parent.parent.joinpath("images").glob(lf.stem + ".*"))
        im = cv2.imread(str(img_p))
        h, w = im.shape[:2]
        for l in lf.read_text().splitlines():
            v = l.split()
            if not v:
                continue
            c, xs = int(v[0]), np.array(v[1:], float)
            col = (0, 0, 255) if c in tgt else (255, 0, 0)
            if len(xs) == 4:
                x, y, bw, bh = xs * [w, h, w, h]
                cv2.rectangle(im, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), col, 3)
                pt = (int(x - bw / 2), int(y - bh / 2) + 25)
            else:
                pts = (xs.reshape(-1, 2) * [w, h]).astype(int)
                cv2.polylines(im, [pts], True, col, 3)
                pt = tuple(pts[0])
            cv2.putText(im, names[c], pt, 0, 0.9, col, 2)
        tiles.append(cv2.resize(im, (400, 400)))
    while len(tiles) % 4:
        tiles.append(np.zeros((400, 400, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out = ROOT / "runs" / "rf_preview" / f"{proj}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"{slug} v{ver}: {len(labels)} label files มี target · grid → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("slugs", nargs="*", default=CANDIDATES)
    p.add_argument("--grid", action="store_true")
    args = p.parse_args()
    key = api_key()
    for slug in args.slugs:
        if args.grid:
            grid(slug, key)
            continue
        r = inspect(slug, key)
        if "error" in r:
            print(f"\n== {slug}: {r['error']}")
            continue
        print(f"\n== {slug} · {r['name']} · {r['type']} · images {r['images']} · license {r['license']} · versions {r['versions']}")
        print(f"   targets: {r['targets'] or '-'}")
        print(f"   classes ({len(r['classes'])}): {', '.join(sorted(r['classes']))[:400]}")


if __name__ == "__main__":
    main()
