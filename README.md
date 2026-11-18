# smart-fridge-cv

ส่วน **Computer Vision** ของ Mini Project *Smart Fridge Recipe Recommender* (ทีม Finoria)
ตรวจจับวัตถุดิบในรูปตู้เย็นด้วย **YOLO11 (Ultralytics)** แล้วส่งรายการวัตถุดิบให้ระบบแนะนำเมนู
repo หลักของทีม: [Finoria-co/fridge-chef-ai](https://github.com/Finoria-co/fridge-chef-ai)

## ใช้งาน (สำหรับคนที่ 5 / 6): `predict_image()`
```bash
pip install ultralytics          # มี torch, opencv, pillow ติดมาด้วย
# ดาวน์โหลด best.pt จาก https://github.com/Finoria-co/smart-fridge-cv/releases/tag/v1.0-yolo11s
# วางไว้ที่ weights/best.pt หรือตั้ง env SMART_FRIDGE_WEIGHTS
```
```python
from predict import predict_image

items = predict_image("fridge.jpg")            # หรือ bytes ของไฟล์ที่อัปโหลด / PIL.Image / numpy (BGR)
ingredients = [it["ingredient"] for it in items]   # เช่น ["chicken", "tomato", "cheese", "bacon"]
```
ผลลัพธ์ (เรียงตาม confidence, ไม่เจออะไร = `[]`):
```json
[{"ingredient": "tomato", "name_th": "มะเขือเทศ", "in_contract": true, "confidence": 0.924, "count": 1,
  "boxes": [[319.3, 119.1, 458.6, 258.9]]}]
```
- `ingredient` ใช้ชื่อตาม **Data Contract 12 ชนิด** ของกลุ่ม: pork, chicken, shrimp, bacon, egg, rice, pasta, bread, holy_basil, tomato, cheese, cabbage (ตาราง `ingredient_map.json`: bacon = เบคอน+แฮม, cheese = cheese butter+cream cheese, chicken = chicken duck)
- ⚠️ **rice, pasta ไม่มีในรูปที่ใช้เทรน → ตรวจจากรูปไม่ได้** ต้องรับจากข้อความพิมพ์ (คนที่ 4)
- `only_contract=False` คืนวัตถุดิบอื่นที่โมเดลรู้จักด้วย (รวม 75 class เช่น `carrot`, `spring_onion`: `in_contract=False`, `name_th=None`)
- `count` = จำนวนชิ้นที่เจอ · `boxes` = กรอบ [x1, y1, x2, y2] (pixel ของรูปต้นฉบับ) ไว้วาดบนหน้าเว็บ · `conf` ค่าเริ่มต้น 0.28 (เลือกจาก F1 curve บน val)
- โหลดโมเดลครั้งแรกช้า (~5 วินาที) ครั้งต่อไปเร็ว (~20–110 ms/รูป บน RTX 4050) → ใน Flask/FastAPI ให้ import ไว้ตอนเริ่ม server
- ลองจาก command line: `python predict.py รูป.jpg [--all] [--conf 0.4]`

| ไฟล์ | เนื้อหา |
|---|---|
| `predict.py`, `ingredient_map.json` | `predict_image()` + ตาราง class → วัตถุดิบมาตรฐาน 12 ชนิด · ทดสอบ: `python tests/test_predict.py` |
| `docs/CV_handoff_guide.docx` | **คู่มือส่งต่องานสำหรับเพื่อน** (ใครทำอะไรต่อ, ใช้ไฟล์ไหน, commit, วิธีเชื่อม predict_image) · สร้างใหม่ด้วย `node docs/make_handoff.js <out.docx>` |
| `PLAN.md` | แผนงานทั้งหมด (Phase A–G) + การโยงกับบทที่เรียน |
| `experiments.md` | ผลการทดลองทุกครั้ง (ตัวเลขจากการรันจริง) |
| `results/<name>/` | `results.csv` และกราฟของแต่ละการทดลอง |
| `data/` | `data.yaml` (75 class), `split_report.md`, รายชื่อรูปใน test |
| `scripts/train.py` | สคริปต์เทรน (ใช้ได้ทั้งในเครื่องและ Colab) |
| `notebooks/colab_train.ipynb` | เทรนบน Google Colab (L4 GPU) |

## Dataset
`foodseg_fridge_thai_yolo` 75 class · train 10,055 / val 1,959 / test 1,317 (test แบ่งจาก val ด้วย `scripts/make_test_split.py`, seed 42)
ตัวรูปไม่อยู่ใน git: ไฟล์ `foodseg_yolo_split.zip` อยู่ใน Google Drive ของทีม (`Mini_Project_Finoria_ML/`)

## เทรนเอง
**Colab**: เปิด `notebooks/colab_train.ipynb` แล้วทำตามเซลล์แรก

**เครื่องตัวเอง** (ต้องมี NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# แตก foodseg_yolo_split.zip แล้ว
python scripts/train.py --name <ชื่อการทดลอง> --data <ที่แตก>/foodseg_yolo/data.yaml --batch 16
```
ถ้าจะเอาผลมาเทียบกับการทดลองใน `experiments.md` ให้ใช้ batch / imgsz / epochs / seed เดียวกัน
