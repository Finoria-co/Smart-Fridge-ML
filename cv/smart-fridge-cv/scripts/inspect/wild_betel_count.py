# เทียบจำนวนกล่องที่ทาย (conf>=0.25) กับจำนวน label ของชะพลู (id 74) ในรูป val ที่มีชะพลู
from pathlib import Path
from ultralytics import YOLO
root = Path("/root/data/foodseg_yolo")
imgs = []
for lf in (root / "labels" / "val").glob("*.txt"):
    n = sum(1 for l in lf.read_text().splitlines() if l.split() and l.split()[0] == "74")
    if n:
        img = next((root / "images" / "val").glob(lf.stem + ".*"))
        imgs.append((img, n))
model = YOLO("runs/detect/baseline_n/weights/best.pt")
tot_lab = tot_pred = 0
rows = []
for img, n in imgs:
    r = model.predict(img, conf=0.25, verbose=False)[0]
    k = int((r.boxes.cls == 74).sum())
    tot_lab += n; tot_pred += k; rows.append((img.name[:40], n, k))
print(f"images {len(imgs)} · labels {tot_lab} · predicted boxes (conf>=0.25) {tot_pred}")
print("prefix of file names:", sorted({r[0].split('_')[0] for r in rows}))
for r in sorted(rows, key=lambda r: r[2] - r[1], reverse=True)[:8]:
    print(r)
