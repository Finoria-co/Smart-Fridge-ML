# smart-fridge-cv

ส่วน **Computer Vision** ของ Mini Project *Smart Fridge Recipe Recommender* (ทีม Finoria)
ตรวจจับวัตถุดิบในรูปตู้เย็นด้วย **YOLO11 (Ultralytics)** แล้วส่งรายการวัตถุดิบให้ระบบแนะนำเมนู
repo หลักของทีม: [Finoria-co/fridge-chef-ai](https://github.com/Finoria-co/fridge-chef-ai)

| ไฟล์ | เนื้อหา |
|---|---|
| `PLAN.md` | แผนงานทั้งหมด (Phase A–G) + การโยงกับบทที่เรียน + โครงคลิปนำเสนอ |
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
