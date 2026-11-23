"""วาดผลโมเดลจริงบนรูป demo ที่คัดไว้ และพิมพ์ผลแบบที่เว็บรวมเรียก (conf 0.25, only_contract=False)
    (WSL) python scripts/demo_render.py <ชื่อไฟล์รูปใน test> ...
ผล: results/demo_pick/images/ (ไม่เข้า git: รูปเป็นของ dataset ภายนอก)
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict import DEFAULT_CONF, load_model, predict_image  # noqa: E402

DATA = Path("/root/data/foodseg_yolo_v2/images/test")
out = ROOT / "results" / "demo_pick" / "images"
out.mkdir(parents=True, exist_ok=True)
model = load_model()
for i, name in enumerate(sys.argv[1:], 1):
    p = DATA / name
    shutil.copy(p, out / f"demo{i}_original{p.suffix}")
    r = model.predict(str(p), conf=DEFAULT_CONF, verbose=False)[0]
    r.save(filename=str(out / f"demo{i}_pred.jpg"))
    contract = [(it["name_th"], it["count"], it["confidence"]) for it in predict_image(str(p))]
    web = [(it["ingredient"], it["count"], it["confidence"]) for it in predict_image(str(p), conf=0.25, only_contract=False)]
    print(f"demo{i}: {name}\n  contract@{DEFAULT_CONF}: {contract}\n  web(all@0.25): {web}")
