import glob, random, cv2, numpy as np
from pathlib import Path
d=Path("/root/data/foodseg_yolo_v2"); CID=40
hits=[]
for split in ["train","val"]:
    for lf in (d/"labels"/split).glob("*.txt"):
        rows=[l.split() for l in lf.read_text().splitlines() if l.strip()]
        if any(int(r[0])==CID for r in rows): hits.append((split,lf,rows))
print("images with cabbage:",len(hits))
import collections; print(collections.Counter(h[1].stem.split('_')[0][:12] for h in hits).most_common(8))
random.seed(0); pick=random.sample(hits,min(12,len(hits)))
tiles=[]
for split,lf,rows in pick:
    ip=next((d/"images"/split).glob(lf.stem+".*")); im=cv2.imread(str(ip)); h,w=im.shape[:2]
    for r in rows:
        if int(r[0])!=CID: continue
        pts=np.array(list(map(float,r[1:]))).reshape(-1,2)*[w,h]
        cv2.polylines(im,[pts.astype(np.int32)],True,(0,0,255),max(2,w//200))
    tiles.append(cv2.resize(im,(320,320)))
while len(tiles)%4: tiles.append(np.zeros((320,320,3),np.uint8))
grid=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
cv2.imwrite("/mnt/c/Users/patim/AppData/Local/Temp/claude/bda2507d44453751/4fdc0140-0185-42a1-92bf-f89f788121d5/scratchpad/cabbage_grid.jpg",grid)
