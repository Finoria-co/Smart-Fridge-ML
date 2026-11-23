"""predict_image(): รับรูปวัตถุดิบ/ตู้เย็น -> รายการวัตถุดิบมาตรฐานที่ตรวจพบ (งานคนที่ 2 ส่งให้คนที่ 5/6)

    from predict import predict_image
    items = predict_image("fridge.jpg")
    # [{"ingredient": "egg", "name_th": "ไข่ไก่", "in_contract": True, "confidence": 0.91, "count": 3,
    #   "boxes": [[x1, y1, x2, y2], ...]}, ...]
    names = [it["ingredient"] for it in items]   # ["egg", "pork", ...] ส่งต่อให้โมเดลจับคู่เมนู

    python predict.py fridge.jpg [--conf 0.25] [--all] [--weights path/to/best.pt]

- image รับได้: path (str/Path), bytes (ไฟล์ที่อัปโหลดจากเว็บ), PIL.Image, numpy array (BGR แบบ cv2.imread)
- ค่าเริ่มต้นคืนเฉพาะ 12 ชนิดของ Data Contract (ingredient_map.json) · only_contract=False คืน class อื่นด้วย
  (ingredient = ชื่อ class เช่น "spring_onion", name_th = None, in_contract = False)
- rice / pasta ไม่มีใน dataset จึงไม่มีทางถูกตรวจจากรูป (ใช้จากข้อความพิมพ์แทน)
- weights: ค่าเริ่มต้น weights/best.pt ข้างไฟล์นี้ หรือตั้ง env SMART_FRIDGE_WEIGHTS
"""
from __future__ import annotations

import argparse
import io
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"
MAP_FILE = ROOT / "ingredient_map.json"
DEFAULT_CONF = 0.28  # เลือกจาก F1-confidence curve บน val ของ yolo11s_v2: F1 เฉลี่ย 12 class ของ Data Contract สูงสุดที่ 0.282 (results/final/conf_from_val.csv, บท 08)
MERGE_IOU = 0.7      # กล่องของ class ต่างกันที่รวมเป็นวัตถุดิบเดียวกัน (เช่น bacon + ham) ทับกันเกินนี้ = ชิ้นเดียวกัน


@lru_cache(maxsize=1)
def load_ingredient_map() -> tuple[dict, dict]:
    """คืน (contract, class_to_std): contract = std -> {"th", "classes"} · class_to_std = ชื่อ class ของโมเดล -> std"""
    contract = json.loads(MAP_FILE.read_text(encoding="utf-8"))["contract"]
    class_to_std = {c: std for std, v in contract.items() for c in v["classes"]}
    return contract, class_to_std


@lru_cache(maxsize=2)
def load_model(weights: str | None = None):
    """โหลดโมเดลครั้งเดียวแล้วใช้ซ้ำ (เรียก predict_image หลายครั้งไม่โหลดใหม่)"""
    from ultralytics import YOLO

    path = Path(weights or os.environ.get("SMART_FRIDGE_WEIGHTS") or DEFAULT_WEIGHTS)
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ weights: {path} · ดาวน์โหลด best.pt จาก GitHub Release ของ repo นี้ "
            f"มาไว้ที่ {DEFAULT_WEIGHTS} หรือตั้ง env SMART_FRIDGE_WEIGHTS")
    return YOLO(str(path))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def aggregate(detections, only_contract: bool = True) -> list[dict]:
    """รวมกล่องที่ตรวจพบเป็นรายการวัตถุดิบ (แยกจากโมเดลเพื่อให้ทดสอบได้โดยไม่ต้องมี GPU/torch)

    detections: iterable ของ (class_name, confidence, [x1, y1, x2, y2])
    """
    contract, class_to_std = load_ingredient_map()
    groups: dict[str, dict] = {}
    for name, conf, box in sorted(detections, key=lambda d: -d[1]):  # conf สูงก่อน
        std = class_to_std.get(name)
        if std is None and only_contract:
            continue
        key = std or name.replace(" ", "_")
        g = groups.setdefault(key, {
            "ingredient": key,
            "name_th": contract[std]["th"] if std else None,
            "in_contract": std is not None,
            "confidence": 0.0,
            "count": 0,
            "boxes": [],
        })
        # Ultralytics ทำ NMS แยกตาม class แล้ว แต่ 2 class ที่รวมเป็นวัตถุดิบเดียวกันอาจทายชิ้นเดียวกันซ้ำ
        if any(_iou(box, b) > MERGE_IOU for b in g["boxes"]):
            continue
        g["boxes"].append([round(float(v), 1) for v in box])
        g["count"] += 1
        g["confidence"] = max(g["confidence"], round(float(conf), 3))
    return sorted(groups.values(), key=lambda g: (-g["confidence"], g["ingredient"]))


def _to_source(image):
    if isinstance(image, (bytes, bytearray)):
        from PIL import Image
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, Path):
        return str(image)
    return image  # str path / PIL.Image / numpy (BGR) ส่งให้ Ultralytics ตรงๆ


def predict_image(image, conf: float = DEFAULT_CONF, only_contract: bool = True,
                  weights: str | Path | None = None, imgsz: int = 640) -> list[dict]:
    """ตรวจจับวัตถุดิบในรูป -> list ของ dict เรียงตาม confidence มากไปน้อย (ไม่เจออะไร = [])"""
    model = load_model(str(weights) if weights else None)
    r = model.predict(_to_source(image), conf=conf, imgsz=imgsz, verbose=False)[0]
    b = r.boxes
    dets = [(r.names[int(c)], s, box) for c, s, box in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
    return aggregate(dets, only_contract=only_contract)


def main():
    p = argparse.ArgumentParser(description="ตรวจวัตถุดิบในรูป แล้วพิมพ์ผลเป็น JSON")
    p.add_argument("images", nargs="+")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--all", action="store_true", help="คืน class นอก Data Contract ด้วย")
    p.add_argument("--weights")
    args = p.parse_args()
    for img in args.images:
        items = predict_image(img, conf=args.conf, only_contract=not args.all, weights=args.weights)
        print(json.dumps({"image": img, "items": items}, ensure_ascii=False))


if __name__ == "__main__":
    main()
