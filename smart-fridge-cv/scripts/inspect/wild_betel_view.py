from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
root = Path("/root/data/foodseg_yolo")
img_p = next((root/"images"/"val").glob("5-thai-herbs_valid_IMG_7004_640_JPG*"))
img = cv2.imread(str(img_p)); h, w = img.shape[:2]
gt = img.copy()
for l in (root/"labels"/"val"/(img_p.stem+".txt")).read_text().splitlines():
    v = l.split()
    pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(int)
    col = (0,0,255) if v[0] == "74" else (255,0,0)
    cv2.polylines(gt, [pts], True, col, 3)
r = YOLO("runs/detect/baseline_n/weights/best.pt").predict(img, conf=0.25, verbose=False)[0]
pr = r.plot(labels=False, conf=False)
both = np.hstack([gt, pr])
cv2.putText(both, "LABEL (red = wild betel)", (10, 30), 0, 1, (0,255,255), 2)
cv2.putText(both, "PREDICTION conf>=0.25", (w+10, 30), 0, 1, (0,255,255), 2)
cv2.imwrite("results/baseline_n/example_wild_betel_label_vs_pred.jpg", both)
print(img.shape)
