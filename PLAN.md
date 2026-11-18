# แผนงาน Computer Vision: Smart Fridge (YOLO) จนถึงคลิปนำเสนอ

เจ้าของงาน: ปฏิมา ถิ่นสำราญ (T-Patima) · โรล **Computer Vision** · เขียนแผน 2026-11-17
ภาพรวมโปรเจกต์และผลตรวจ dataset: `../mini-project-smart-fridge.md` (อ่านก่อน)

**เป้าหมาย**: เทรน **YOLO (Ultralytics YOLO11)** ตรวจจับวัตถุดิบในรูปตู้เย็น แล้วส่งมอบ
`best.pt` + `predict_image()` + กราฟ/ตารางผล + ส่วน CV ของคลิปนำเสนอ
## ⭐ ขอบเขตงานจริง (ปรับ 2026-11-18 ตามแผนกลุ่ม `../Smart_Fridge_ML_Project_Plan.pdf`)
เราคือ **คนที่ 2: Computer Vision ML** (ไม่ได้ทำทั้งโปรเจกต์) หน้าที่ตามแผนกลุ่ม:
- [x] เทรน YOLO บน Colab (`baseline_n`, `baseline_n_v2`)
- [x] วัดผล mAP, Confusion Matrix, Loss Curve (`results/`, `experiments.md`) · **test (ครั้งเดียว): mAP50 0.643 / mAP50-95 0.525** (`results/final/`)
- [x] เลือกโมเดลสุดท้าย: **YOLO11s (`yolo11s_v2`)** val v2 mAP50-95 0.486 vs n 0.436 · ความเร็วเท่ากัน (`experiments.md` #3)
- [ ] ส่ง **`best.pt`** (GitHub Release) หลังวัดผล test (Phase E)
- [x] **`predict_image()`** คืนชื่อวัตถุดิบมาตรฐานตาม Data Contract 12 ชนิดของกลุ่ม (`predict.py`, `ingredient_map.json`, `tests/test_predict.py`) · 2026-11-18 ทดสอบกับ `baseline_n_v2` แล้ว · เหลือ: เปลี่ยน weights เป็นตัวสุดท้าย + เลือก conf จาก F1 curve + ให้คนที่ 6 ลองเรียก
- [ ] **สไลด์หน้า 5**: การเทรนโมเดลตรวจจับรูปภาพ + กราฟวัดผล (คนที่ 6 รวมสไลด์)

ไม่ใช่งานเรา (ตัดออก): สำรวจ/เตรียม dataset และสไลด์หน้า 4 = คนที่ 1 · รวมสไลด์ + Live Demo = คนที่ 6 · คลิปทั้งเรื่อง (Phase G เดิม) · GA/TTA/seg/k-NN-SVM เป็นของแถม ไม่อยู่ในแผนกลุ่ม
Data Contract 12 ชนิด: pork, chicken, shrimp, bacon (เบคอน/แฮม), egg, rice, pasta, bread, holy_basil, tomato, cheese, cabbage · **rice, pasta ไม่มีใน dataset → ตรวจจากรูปไม่ได้** (แจ้งคนที่ 4/6: ใช้จากข้อความพิมพ์)

**ข้อบังคับ**: ใช้ YOLO · ดึงเนื้อหาที่เรียนในวิชามาใช้และอ้างบทในสไลด์ทุกครั้ง (ส่วนที่ไม่มีในสไลด์ให้ติดป้าย "เนื้อหาเสริม")

---

## 0. สถานะตอนเริ่ม (2026-11-17)
- [x] ตรวจแผน PDF + แผนผังล่าสุด + dataset → `../mini-project-smart-fridge.md`
- [x] แตก dataset ไว้ที่ `data/foodseg_yolo/` (1.9 GB, gitignore) และแบ่ง test แล้ว: train 10,055 / val 1,959 / test 1,317 (`data/split_report.md`)
- [x] `data/data.yaml` มี `test:` แล้ว และไม่มี `path:` (Ultralytics จะใช้โฟลเดอร์ของไฟล์ yaml)
- [ ] **รอลงมติทีม (ข้อ 4.1 ใน mini-project-smart-fridge.md)**: เทรน 75 class หรือตัดเหลือ 12 ชนิด (dataset ไม่มีข้าวและพาสต้า)
  - ถ้ายังไม่ได้มติ ให้ใช้ค่าเริ่มต้น **75 class** แล้วแปลงเป็นวัตถุดิบมาตรฐานด้วยตาราง (เปลี่ยนเป็น 12 ทีหลังได้ด้วยสคริปต์กรอง class)
- เครื่อง: Windows 11, **GPU RTX 4050 Laptop 6 GB** (driver 581.66) → เทรนในเครื่องได้ · Colab เป็นทางสำรอง

---

## 1. ผูกเนื้อหาที่เรียน (บทในสไลด์) เข้ากับงาน CV
ใช้ตารางนี้ทั้งตอนทำและตอนเขียนสคริปต์คลิป ทุกข้อต้องมีหลักฐาน (กราฟ/ตาราง/ตัวเลข) ไว้โชว์

| บท | เนื้อหาในสไลด์ | ใช้ในงาน CV ยังไง | หลักฐานที่ต้องมี |
|---|---|---|---|
| 01 Intro | นิยาม ML: Task T, Performance P, Experience E | T = ตรวจจับวัตถุดิบในรูปตู้เย็น · P = mAP50-95 บน test · E = รูปที่ label แล้ว 10,055 รูป | สไลด์ Problem |
| 05 ANN | perceptron, activation, gradient descent, learning rate, backpropagation | YOLO คือ ANN หลายชั้น เทรนด้วย gradient descent (SGD/AdamW) + backprop ลด loss · อธิบาย `lr0`, `epochs`, `batch` · CNN เป็นเนื้อหาเสริมต่อจาก ANN | loss curve (box/cls/dfl) ของ train vs val |
| 03 Decision Tree | overfitting, validation set, pre-pruning / post-pruning | ใช้ val ตรวจ overfitting (train loss ลด แต่ val loss เพิ่ม) · `patience` (early stopping) คือแนวคิดเดียวกับ pre-pruning · เลือก `best.pt` จาก val แล้วใช้ test **ครั้งเดียวตอนจบ** | กราฟ train vs val + epoch ที่หยุด |
| 07 GA | fitness, selection, crossover, mutation | **ปรับ hyperparameter ด้วย evolution**: `model.tune()` ของ Ultralytics กลายพันธุ์ค่า hyperparameter แล้วคัดตัวที่ fitness (mAP) ดีที่สุด (ของ Ultralytics ใช้ mutation + เลือกตัวดี ไม่มี crossover ให้บอกตรงๆ) · โยงกับการบ้าน GA จัดทีมกีฬาสีได้ | กราฟ fitness ตามรุ่น + ค่า hyperparameter ที่ได้ |
| Ensemble (Classwork) | Bagging/Boosting: รวมหลายโมเดลลด variance/bias | **TTA** (`augment=True`) = ทำนายหลายมุมแล้วรวม (แนวคิดเดียวกับ Bagging) · (ทางเลือก) รวมผล YOLO11n + YOLO11s ด้วย Weighted Boxes Fusion | ตาราง mAP ก่อน/หลัง TTA |
| 04 k-NN | k-NN, การวัดระยะ L2, Condensed NN | **ตัวจำแนกขั้นที่ 2 สำหรับสมุนไพรที่สับสนกัน** (กะเพรา/โหระพา/มะกรูด/ชะพลู): crop กล่องจาก YOLO → embedding → k-NN (L2) · Condensed NN ลดจำนวนตัวอย่างที่เก็บ | confusion matrix ก่อน/หลังใช้ k-NN เฉพาะกลุ่มนี้ |
| 09 SVM | hyperplane, margin, kernel | เทียบกับ k-NN บน embedding เดียวกัน (linear vs RBF kernel) เป็นหัวข้อ Compared | ตาราง accuracy k-NN vs SVM |
| 06 Clustering | k-means, SOM | สำรวจ dataset: k-means บน embedding ของรูป เพื่อดูกลุ่ม/รูปซ้ำ/รูปผิดปกติ (ใช้เป็นส่วน data ของ Method) | กราฟ cluster 2D + ตัวอย่างรูปแต่ละกลุ่ม |
| 08 Bayesian | Bayes theorem, MAP | เลือก confidence threshold: เทียบ precision/recall ที่ threshold ต่างๆ (precision/recall/mAP เป็นเนื้อหาเสริม) · class imbalance = prior ไม่เท่ากัน | กราฟ PR curve / F1–confidence (Ultralytics สร้างให้) |

> แกนบังคับ: บท 01, 05, 03, 07 + TTA · ทางเลือกถ้ามีเวลา: k-NN/SVM ขั้นที่ 2, k-means สำรวจข้อมูล, WBF ensemble

---

## 2. Phase ทำงาน (commit ทีละขั้นทุกครั้ง)

### Phase A: เตรียมเครื่อง ✅ เสร็จ 2026-11-17 (ย้ายไป WSL2)
> **ผลจริง**: venv บน Windows ติดตั้ง torch ได้ แต่ Smart App Control (state = On) บล็อก `torch\lib\torch.dll` ทุกครั้ง (ดูจาก event log CodeIntegrity, ลองทั้ง 2.11.0 และ 2.9.1) ผู้ใช้จึงเลือกใช้ **WSL2 Ubuntu** แทน
> venv `/root/venvs/smart-fridge` (Python 3.12, torch 2.11.0+cu128, ultralytics 8.4.215, `cuda.is_available()` = True) · สร้างด้วย `scripts/setup_wsl.sh` · dataset คัดลอกไปไว้ที่ `/root/data/foodseg_yolo/` (นับไฟล์แล้วตรงทุก split) · วิธีเรียกใช้ดูใน `CLAUDE.md`
>
> ขั้นตอนเดิม (ใช้ไม่ได้บนเครื่องนี้ เก็บไว้อ้างอิง):

1. สร้าง venv **ด้วย `C:\Users\patim\miniconda3\python.exe -m venv .venv`** (ห้าม `uv venv`: Application Control บล็อก)
2. ติดตั้ง PyTorch แบบ CUDA ก่อน แล้วค่อย `ultralytics` (เช็คคำสั่งติดตั้งปัจจุบันจาก pytorch.org ให้ตรงกับ Python ใน venv ถ้า Python 3.14 ยังไม่มี wheel ของ torch ให้ใช้ Python 3.12 จาก conda env แยก)
3. ตรวจ: `python -c "import torch; print(torch.cuda.is_available())"` ต้องได้ `True`
4. ถ้าเจอ `DLL load failed ... Application Control` ให้รันซ้ำ ใช้ `python -m <module>` แทนตัว `.exe` เสมอ (เช่น `python -m ultralytics ...` หรือเรียกผ่าน Python API)
5. `requirements.txt` และ commit

### Phase B: ตารางวัตถุดิบมาตรฐาน + สำรวจข้อมูล · ✂️ ตัดออก 2026-11-18 (สำรวจ dataset = คนที่ 1 · ตารางวัตถุดิบย้ายไปทำใน Phase F เป็น `ingredient_map.json` ตาม Data Contract 12 ชนิด)
1. `ingredient_map.json`: class id → `{"en": ..., "th": ..., "std": ...}` (ชื่อมาตรฐานต้องตรงกับฝ่าย NLP/Retrieval) · ถ้าทีมเลือก 12 ชนิด ให้เขียน `scripts/filter_classes.py` สร้าง dataset ย่อย
2. notebook `01_data.ipynb`: จำนวนรูป/instance ต่อ class (กราฟ), ขนาดรูป, ตัวอย่างรูปพร้อม polygon, class imbalance
3. (ทางเลือก บท 06) k-means บน embedding ของรูปเพื่อดูกลุ่มและหารูปผิดปกติ

### Phase C: Baseline ✅ เสร็จ 2026-11-17 (Colab L4, mAP50 0.510 / mAP50-95 0.409 บน val · ดู experiments.md #1)
1. ทดลองสั้นก่อน: `yolo11n.pt`, `epochs=1`, `fraction=0.1` เพื่อตรวจว่า pipeline ทำงาน
2. Baseline: **YOLO11n detect**, `imgsz=640`, `epochs=50`, `patience=15`, `batch=16` (ถ้า GPU memory ไม่พอใช้ `batch=8` หรือ `batch=-1` ให้ Ultralytics เลือกเอง), `workers=2` (Windows), `seed=0`
   - label เป็น polygon: Ultralytics แปลงเป็นกล่องให้เองตอนเทรน detect
3. เก็บผล: `results.csv`, loss curve, PR curve, confusion matrix, mAP50 / mAP50-95 ต่อ class บน **val**
4. วิเคราะห์ overfitting (บท 03) และ class ที่แย่สุด (คาดว่าสมุนไพรไทยและ class รูปน้อย)

### Phase D: ปรับปรุง + Compared (~1 วัน รวมเวลาเทรน)
- ✅ **0. Dataset v2 (บท 01 E)** 2026-11-18: เพิ่มวัตถุดิบดิบ 709 รูป → `baseline_n_v2` val ดิบ mAP50-95 0.392 → 0.674, val เดิมไม่ลด (0.407 → 0.417) · รายละเอียด `experiments.md` #2 · **การทดลองต่อจากนี้ใช้ dataset v2**

เลือกตามเวลาที่มี เรียงตามความสำคัญ (2026-11-18: ทำแค่ข้อ 1 · ข้อ 2–5 ไม่อยู่ในแผนกลุ่ม ทำเฉพาะถ้ามีเวลาเหลือ):
1. ✅ **YOLO11s** (`yolo11s_v2`) ดีกว่า n ทุก val · เลือกเป็นโมเดลสุดท้าย (`experiments.md` #3) เทียบกับ n (ขนาดโมเดล vs ความแม่นยำ vs ความเร็ว)
2. **GA tuning (บท 07)**: `model.tune(data=..., epochs=10, iterations=20–30, fraction=0.3, plots=True)` แล้วเทรนเต็มด้วย hyperparameter ที่ได้ ระวังเวลา: ตั้ง iteration ให้จบได้ภายในคืนเดียว
3. **TTA (Ensemble)**: `model.val(..., augment=True)` เทียบกับไม่ใช้
4. (ทางเลือก) **YOLO11n-seg** เทียบกับ detect: label รองรับอยู่แล้ว
5. (ทางเลือก บท 04/09) ตัวจำแนกขั้นที่ 2 สำหรับสมุนไพร: crop → embedding (เช่น feature จาก backbone ของ YOLO หรือ color histogram) → k-NN vs SVM

ทุกการทดลองบันทึกลง `experiments.md`: config, mAP50, mAP50-95, precision, recall, เวลาเทรน, ms/รูป

### Phase E: ประเมินผลสุดท้ายบน test ✅ 2026-11-18 (yolo11s_v2 · conf 0.28 จาก val · test mAP50-95 0.525 · `experiments.md` Phase E) · ข้อ 3 รูปตู้เย็นจริง = ทางเลือก ใช้ร่วมกับ Live Demo ของคนที่ 6
1. เลือกโมเดลสุดท้ายจากผลบน **val** เท่านั้น
2. รัน `model.val(split="test")` **ครั้งเดียว** เก็บตาราง mAP ต่อ class + confusion matrix ของ test
3. ทดสอบกับรูปถ่ายตู้เย็นจริง 10–20 รูป (ไม่อยู่ใน dataset) บันทึกว่าถูก/ผิดอะไร: ใช้เป็นหัวข้อ Discuss และ Live demo

### Phase F: ส่งมอบให้ทีม (~2 ชม.)
1. `predict.py`:
   ```python
   def predict_image(image, conf=0.25) -> list[dict]:
       # [{"ingredient": "egg", "name_th": "ไข่ไก่", "confidence": 0.91, "count": 3}, ...]
   ```
   - รวมกล่องของ class เดียวกันเป็น 1 รายการพร้อม `count`, แปลงด้วย `ingredient_map.json`, เลือก `conf` จาก F1–confidence curve (บท 08)
   - ค่าเริ่มต้นคืนเฉพาะ 12 ชนิดของ Data Contract (`only_contract=True`) · class อื่นคืนได้ถ้าตั้ง False · **รอตกลงกับคนที่ 5/6** ว่าจะใช้แบบไหน
2. `best.pt` (ถ้าใหญ่เกิน 100 MB ห้ามใส่ git ให้ส่งทาง Drive/Release)
3. `README.md`: วิธีติดตั้ง วิธีเรียก `predict_image()` ตัวอย่าง input/output
4. ส่งให้ฝ่าย Integration ลองเรียกจริง แล้วแก้ตาม feedback

### Phase G: สไลด์หน้า 5 (ปรับ 2026-11-18)
แผนกลุ่มให้คนที่ 2 ทำและพูด **สไลด์หน้า 5: Computer Vision Model** (การเทรน + กราฟความแม่นยำ) เท่านั้น: dataset v1 → v2 (เหตุผลบท 01), loss curve (บท 03/05), mAP บน test, confusion matrix, ตารางเทียบ v1/v2 และ n/s · ส่งให้คนที่ 6 รวม

<details><summary>แผนคลิปเดิม (เก็บไว้อ้างอิง ไม่ต้องทำครบ)</summary>

ทำตามลำดับหัวข้อของอาจารย์ **Problem → SOTA → Method → Result → Compared → Discuss → Live demo**

| หัวข้อ | เนื้อหาส่วน CV | ภาพที่ใช้ |
|---|---|---|
| Problem | ทำไมต้องรู้ว่าในตู้เย็นมีอะไรจากรูป · นิยาม T/P/E (บท 01) | รูปตู้เย็นจริง + ผลลัพธ์ที่อยากได้ |
| SOTA | ตระกูล YOLO (v8 → YOLO11), detection vs segmentation, งาน food segmentation (ที่มาของ dataset) | timeline YOLO, ตัวอย่างงานอื่น (อ้างอิงให้ครบ) |
| Method | dataset 75 class / แบ่ง train-val-test / YOLO11 = ANN เทรนด้วย gradient descent + backprop (บท 05) / early stopping (บท 03) / GA tuning (บท 07) | กราฟจำนวนรูปต่อ class, แผนผัง pipeline |
| Result | mAP50, mAP50-95 บน test, loss curve, confusion matrix, class ที่ดี/แย่สุด | กราฟจาก Ultralytics |
| Compared | n vs s, ก่อน/หลัง GA tuning, TTA (ensemble), (detect vs seg), (k-NN vs SVM สำหรับสมุนไพร) | ตารางเดียวรวมทุกการทดลอง |
| Discuss | ข้อจำกัด: ไม่มีข้าว/พาสต้า, class รูปน้อย, กะเพรา vs โหระพาสับสน, val ของสมุนไพรไทยน้อย, รูปตู้เย็นจริงต่างจาก dataset · แนวทางพัฒนาต่อ | confusion matrix ซูมกลุ่มสมุนไพร |
| Live demo | ถ่ายรูปตู้เย็น → `predict_image()` → รายการวัตถุดิบ → (ต่อกับระบบของทีม) เมนู Top-5 | อัดหน้าจอจริง |

- เขียนบทพูด (`video_script.md`) แยกตามหัวข้อ ระบุเวลาต่อหัวข้อ และตัวเลขทุกตัวต้องมาจากผลรันจริง
- อัดสำรองคลิป Live demo ไว้เผื่อ demo สดพัง
- ตรวจว่าทุกบทที่อ้าง (01/03/05/07/...) พูดถึงในคลิปจริง

</details>

---

## 3. กติกาการทำงาน (สำหรับเซสชันใหม่)
- ตอบภาษาไทย อ้างบทในสไลด์ (`../.claude/course-text/`) ส่วนที่เกินสไลด์ติดป้าย "เนื้อหาเสริม"
- **ก่อนเขียนโค้ดงานใหม่ ให้สัมภาษณ์ผู้ใช้ใน plan mode ก่อน** (กติกาจาก memory) โดยเฉพาะมติ 75 vs 12 class
- **commit ทันทีหลังแก้แต่ละขั้น** push / PR / merge เฉพาะเมื่อผู้ใช้สั่ง
- ห้ามแต่งตัวเลข ทุกตัวเลขต้องมาจากการรันจริง
- ใช้ test ครั้งเดียวตอนจบ (Phase E) ห้ามจูนโดยดูผล test
- dataset และไฟล์ `runs/` ห้ามเข้า git
