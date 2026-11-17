# หารูป val ที่มี pork แล้วโมเดลทายเป็น steak (IoU>0.5) วาด label vs prediction 3 รูป
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
root = Path("/root/data/foodseg_yolo"); PORK, STEAK = 12, 11
model = YOLO("runs/detect/baseline_n/weights/best.pt")
picked, n_img = [], 0
for lf in sorted((root/"labels"/"val").glob("*.txt")):
    rows = [l.split() for l in lf.read_text().splitlines() if l.split()]
    pork = [r for r in rows if r[0] == str(PORK)]
    if not pork: continue
    n_img += 1
    img_p = next((root/"images"/"val").glob(lf.stem + ".*")); img = cv2.imread(str(img_p)); h, w = img.shape[:2]
    gtb = []
    for r in pork:
        pts = np.array(r[1:], float).reshape(-1, 2) * [w, h]
        gtb.append([*pts.min(0), *pts.max(0)])
    res = model.predict(img, conf=0.25, verbose=False)[0]
    pb = res.boxes
    if len(pb) == 0: continue
    iou = box_iou(torch.tensor(gtb, dtype=torch.float32), pb.xyxy.cpu())
    hit_steak = ((iou > 0.5) & (pb.cls.cpu() == STEAK)).any()
    if hit_steak and len(picked) < 3:
        gt = img.copy()
        for r in rows:
            pts = (np.array(r[1:], float).reshape(-1, 2) * [w, h]).astype(int)
            cv2.polylines(gt, [pts], True, (0,0,255) if r[0] == str(PORK) else (255,0,0), 3)
        picked.append(np.hstack([gt, res.plot(conf=True, font_size=14, line_width=3)]))
        print("picked", img_p.name[:60])
print("val images with pork:", n_img)
out = np.vstack([cv2.resize(p, (1280, 640)) for p in picked])
cv2.imwrite("results/baseline_n/example_pork_as_steak.jpg", out)
