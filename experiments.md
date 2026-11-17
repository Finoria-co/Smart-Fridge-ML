# บันทึกการทดลอง (CV · YOLO11)

ทุกตัวเลขมาจากการรันจริง · ประเมินผลบน **val** เท่านั้น จนถึง Phase E (ใช้ test ครั้งเดียว)

## สภาพแวดล้อม
| รายการ | ค่า |
|---|---|
| ที่รัน | WSL2 Ubuntu บน Windows 11 (Smart App Control บล็อก torch บน Windows) |
| GPU | NVIDIA GeForce RTX 4050 Laptop 6 GB · driver 581.66 |
| Python / torch | 3.12.16 / 2.11.0+cu128 |
| ultralytics | 8.4.215 |
| dataset | `/root/data/foodseg_yolo` · 75 class · train 10,055 / val 1,959 / test 1,317 |

## ตารางสรุป
| # | วันที่ | โมเดล | config | epochs (หยุดที่) | mAP50 (val) | mAP50-95 (val) | P | R | เวลาเทรน | ms/รูป | บท |
|---|---|---|---|---|---|---|---|---|---|---|---|

| 0 | 2026-11-17 | YOLO11n detect | smoke test (RTX 4050/WSL): `fraction=0.1` (1,006 รูป), batch 16, imgsz 640 | 1 | 0.0217 | 0.0171 | 0.161 | 0.037 | 2.2 นาที | 13.2 | – |
| 1 | 2026-11-17 | YOLO11n detect | **baseline** (Colab L4): batch 32, imgsz 640, patience 15, seed 0, cache ram | 50 (best 46) | **0.5101** | **0.4090** | 0.487 | 0.522 | 46.1 นาที | 1.3 (L4) | 01, 03, 05 |
| 2 | 2026-11-18 | YOLO11n detect | **baseline_n_v2**: config เดียวกับ #1 ทุกค่า เปลี่ยนแค่ dataset → **v2** (+709 รูปวัตถุดิบดิบ) · ⚠️ val v2 (2,068 รูป) ≠ val #1 จึงเทียบตรงไม่ได้ ดูตารางเทียบใน #2 | 50 (best 44) | 0.5558 | 0.4363 | 0.592 | 0.545 | 48.1 นาที | – | 01, 03 |

## รายละเอียดแต่ละการทดลอง

### #0 smoke (`runs/detect/smoke`)
- จุดประสงค์: ตรวจว่า pipeline ทำงาน **ไม่ใช่ผลที่ใช้เทียบ** (เทรนแค่ 1 epoch บน 10% ของ train)
- คำสั่ง: `scripts/train.py --name smoke --epochs 1 --fraction 0.1`
- ผล: สแกน label ได้ไม่มีรูป corrupt · label แบบ polygon แปลงเป็นกล่องเองได้ · ไม่มี warning · AMP ผ่าน
- GPU memory ตอนเทรน 2.49 GB ที่ batch 16 (จาก 6 GB) → baseline ใช้ batch 16 ได้
- ความเร็ว: 63 batch ใช้ 20.8 วินาที (~3.0 it/s) → ประมาณการ 1 epoch เต็ม (629 batch) ≈ 3.5 นาที + val ≈ 40 วินาที (**เป็นการประมาณ** ยังไม่ได้วัดจริง)
- loss epoch 1: train box 1.244 / cls 6.986 / dfl 1.527 · val box 1.161 / cls 5.255 / dfl 1.433

### #1 baseline_n (`results/baseline_n/`)
- คำสั่ง (Colab L4, `notebooks/colab_train.ipynb` เซลล์ 5): `train.py --name baseline_n --model yolo11n.pt --epochs 50 --patience 15 --batch 32 --workers 8 --cache ram`
- เวลา 46.1 นาทีตาม `results.csv` ≈ 1.5 CU (รอบซ้ำใช้ 46.3 นาที, train.py รายงานรวมโหลดข้อมูล 47.4 นาที)
- **ทำซ้ำได้ (reproducible)**: เซลล์ 5 ถูกรันจนจบ 2 รอบ (`baseline_n`, `baseline_n-3`) ได้ P/R/mAP/loss ตรงกันทุกคอลัมน์ทั้ง 50 epoch และ mAP ราย class ตรงกันทุกค่า เพราะ `seed=0` + `deterministic=True` · เก็บไว้แค่ `baseline_n` ใน Drive (ลบ `baseline_n-3` ที่ซ้ำ และ `baseline_n-2` ที่หยุดหลัง epoch 1 · 2026-11-17)
- **best epoch 46** (Ultralytics เลือก best.pt ด้วย fitness = 0.1·mAP50 + 0.9·mAP50-95): P 0.4873 · R 0.5220 · mAP50 0.5101 · mAP50-95 0.4090
  - epoch 50: P 0.5342 · R 0.4898 · mAP50 0.5017 · mAP50-95 0.4028
  - val ซ้ำด้วย best.pt ในเครื่อง (batch 16): mAP50 0.5079 · mAP50-95 0.4073 (ต่างจากตอนเทรนเล็กน้อยเพราะ batch ตอน val ต่างกัน)
- ความเร็ว inference: 1.3 ms/รูป บน L4 · 3.3 ms/รูป บน RTX 4050 (batch 16)

**บท 03 (overfitting / early stopping)** → `loss_train_vs_val.png`
- val loss ลดตลอดแล้วแบนหลัง ~epoch 30 (val box ต่ำสุด epoch 49, cls epoch 40, dfl epoch 29) และ**ไม่กลับขึ้น** → ยังไม่ overfit ชัดเจน
- แต่ช่องห่าง train–val กว้างขึ้นหลัง epoch 30 (train ยังลด val แบน) = โมเดลเริ่มจำ train มากกว่าเรียนรู้สิ่งที่ใช้กับข้อมูลใหม่ได้
- early stopping (patience 15) ไม่ทำงานเพราะ fitness ยังดีขึ้นช้าๆ จนถึง epoch 46
- train loss ตกฮวบที่ epoch 41 = Ultralytics ปิด mosaic augmentation 10 epoch สุดท้าย (`close_mosaic=10`, เนื้อหาเสริม) ไม่ใช่ความผิดปกติ

**ราย class** (`per_class_val.csv`, 73 class: plum และ raspberries ไม่มีใน val)
- แย่สุด (mAP50-95): okra 0.000 · ginger 0.010 · squash 0.063 · salad 0.065 · mango 0.067 · white radish 0.082 · bean sprouts 0.095 · tofu 0.101 · spring onion 0.109 · pumpkin 0.136
- ดีสุด: sweet potato 0.896 · peach 0.863 · melon 0.821 · watermelon 0.756 · apple 0.740 · corn 0.732
- ⚠️ class ที่มี instance ใน val น้อยมาก (sweet potato 1, okra 2, squash 2, peach 17 จาก 2 รูป) ค่า mAP แกว่งได้มาก อย่าตีความเกิน
- จำนวนรูปใน train สัมพันธ์กับ mAP50-95 น้อย (Spearman 0.19): กลุ่ม train < 50 รูป เฉลี่ย 0.391 · 50–200 รูป 0.312 · > 200 รูป 0.475
- วัตถุดิบในแผน 12 ชนิด: egg 0.790 / 0.547 · tomato 0.725 / 0.569 · holy basil 0.652 / 0.551 · bread 0.517 / 0.396 · chicken duck 0.418 / 0.341 · bacon 0.615 / 0.308 (7 instance) · ham 0.331 / 0.205 · shrimp 0.294 / 0.180 · **pork 0.160 / 0.143** (mAP50 / mAP50-95)
- สมุนไพรไทย: sweet basil 0.549 · holy basil 0.551 · kaffir lime 0.553 · wild betel 0.494 (R 0.906 แต่ P 0.293 → ทายเป็นชะพลูบ่อยเกิน ต้องดู confusion matrix)

**สิ่งที่ได้ไปใช้ต่อ (Phase D)**
- pork / shrimp / chicken คือวัตถุดิบหลักของแผนที่ยังแย่ → ดูใน confusion matrix ว่าสับสนกับอะไร (เนื้อสัตว์ด้วยกัน?)
- mAP ยังขึ้นช้าๆ ที่ epoch 50 → YOLO11s และ/หรือเทรนนานขึ้นมีโอกาสดีขึ้น

**วิเคราะห์ความผิดพลาด (confusion matrix + ดูรูปจริง)** → `confusions_val.csv`, `scripts/top_confusions.py`, `scripts/inspect/`
> ⚠️ คอลัมน์ "ทายว่าเป็น X แต่ของจริงคือ background" ใน confusion matrix ของ Ultralytics นับกล่อง conf ต่ำมากด้วย ตัวเลขจึงสูงเกินจริง ใช้ตีความไม่ได้ ใช้เฉพาะฝั่ง "ของจริงคือ X แต่ทายเป็น"
- **pork** (145 instance): ถูก 30 · ทายเป็น steak 46 (32%) · chicken duck 23 (16%) · ไม่เจอ 12 · mushroom 10
  - ดูรูปจริง (`example_pork_as_steak.jpg`): รูป pork ใน val มาจาก**อาหารปรุงสุกบนจาน** (หมูย่างราดซอส, มีทโลฟ, หมูสไลซ์) ไม่ใช่หมูดิบในตู้เย็น · แยกจาก steak ด้วยตาก็ยาก → ปัญหาหลักคือ**ข้อมูลไม่ตรงกับงานจริง (domain mismatch)** มากกว่าโมเดล
- **shrimp** (48): ถูก 19 · chicken duck 8 · fish 3 · **ham** (11): ถูก 4 · cheese butter 3
- **wild betel**: เจอของจริง 179/180 แต่ precision 0.293 เพราะ**label ไม่ครบ**: ใน 44 รูป val มี label 180 กล่อง แต่โมเดลทาย 633 กล่อง (conf ≥ 0.25) · ตัวอย่าง IMG_7004 มีใบราว 30 ใบ label ไว้ 2 ใบ โมเดลเจอเกือบทุกใบ (`example_wild_betel_label_vs_pred.jpg`) → precision ต่ำเพราะ label ไม่ใช่เพราะโมเดล · รูปทั้งหมดมาจาก `5-thai-herbs` = ต้นชะพลูในสวน ไม่ใช่ใบในตู้เย็น
- **mango** (12): ทายเป็น apple 3, cheese butter 2, carrot 2 · **ginger** (10): ถูก 1 · **okra** (2): ถูก 0 → รูปใน train น้อย (15–73 รูป) และใน val มีไม่กี่ instance

## Dataset v2: เพิ่มวัตถุดิบดิบจาก Roboflow (2026-11-17)
- เหตุผล: pork ใน v1 เป็นอาหารสุกบนจาน (#1) → E ไม่ตรงกับ T (บท 01 *"Is training experience representative of performance goal?"*)
- ที่มา 4 ชุด (`data/sources.md`): aicook v1 (MIT), fridge-objects v9, food-ingredients-x2 v1, meat-data-sets v4 (CC BY 4.0) · map class: `data/rf_class_map.yaml` (เนื้อวัวดิบ → steak, รูปที่มี class map ไม่ได้ = ตัดทั้งรูป)
- คัดรูป (`scripts/build_dataset_v2.py`): ต้องมี pork/shrimp/ham/bacon หรือเนื้อดิบที่สับสนกับ pork (steak, chicken duck) · ตัดรูปซ้ำกับ v1 76 รูป · 1 รูปต่อต้นฉบับ · จำกัด 300 รูป/class → **เพิ่ม 709 รูป** แบ่ง 70/15/15 (seed 42) → train 10,545 / val 2,068 / test 1,427 (`data/split_report_v2.md`)
- เพิ่มต่อ class (train/val/test): pork +119/+26/+26 · shrimp +129/+28/+33 · ham +55/+11/+12 · bacon +63/+14/+13 · steak +133/+29/+29 · chicken duck +203/+44/+43 · class อื่นในรูปตู้เย็น aicook ก็เพิ่มด้วย (เช่น sugar +114, flour +105, sweet potato +94)
- label: Roboflow เป็นกล่อง → แปลงเป็น polygon ด้วย SAM (sam_b) · ใช้ polygon ถ้ากรอบของมันทับกล่องเดิม IoU ≥ 0.8 (3,405 ชิ้น) ไม่งั้นใช้สี่เหลี่ยมของกล่องเดิม (1,199 ชิ้น) → กรอบสำหรับ detect ตรงกับต้นฉบับ
  - ⚠️ ข้อจำกัด: polygon จาก SAM บางชิ้นใน food-ingredients-x2 เป็นเส้นซิกแซก (SAM ได้หลาย mask แล้วถูกต่อกัน) กรอบยังถูกแต่ไม่เหมาะกับงาน segmentation (`results/dataset_v2/v2_added_food-ingredients-x2.jpg`)
- Ultralytics สแกน label v2: 0 corrupt

**baseline_n เดิม บน val ส่วนที่เพิ่ม (109 รูปวัตถุดิบดิบ)** → `results/baseline_n_on_v2added/per_class_val.csv` (ค่า "ก่อน")
- all: mAP50 0.6220 · mAP50-95 0.3924
- pork mAP50 0.166 / mAP50-95 0.114 (R 0.077: เจอหมูดิบแค่ 2 จาก 26) · bacon 0.000 / 0.000 · ham 0.555 / 0.383 · shrimp 0.787 / 0.402 · steak 0.484 / 0.268 · chicken duck 0.600 / 0.474

### #2 baseline_n_v2 (`results/baseline_n_v2/`, 2026-11-18)
- คำสั่ง: เซลล์ 5 ของ notebook · `--name baseline_n_v2 --data /content/foodseg_yolo_v2/data.yaml` ที่เหลือเหมือน #1 ทุกค่า (ตรวจด้วย diff `args.yaml`: ต่างแค่ `data`, `name`)
- ใน Drive ชื่อ `baseline_n_v2-2`: รอบแรก `baseline_n_v2` session หลุดหลัง epoch 13 (ไม่ใช้) · รอบที่ใช้ copy มาเป็น `runs/detect/baseline_n_v2`
- Colab L4 · torch 2.14.1+cu130 · ultralytics 8.4.215 · เวลา 48.1 นาทีตาม `results.csv`
- best epoch 44 (บน val v2): P 0.592 · R 0.545 · mAP50 0.5558 · mAP50-95 0.4363 · epoch 50: mAP50 0.5528 · mAP50-95 0.4339
- บท 03: val loss แบนช่วงท้ายและไม่กลับขึ้น (box ต่ำสุด epoch 49, cls epoch 34, dfl epoch 42) → ไม่ overfit ชัดเจน เหมือน #1 (`loss_train_vs_val.png`)

**เทียบ #1 (v1) กับ #2 (v2) บน val ชุดเดียวกัน** (eval ในเครื่อง batch 16 · `eval_per_class.py --data ...`)

| val | รูป | #1 mAP50 / mAP50-95 | #2 mAP50 / mAP50-95 |
|---|---|---|---|
| val เดิม v1 (`results/baseline_n_v2_on_v1val/`) | 1,959 | 0.508 / 0.407 | 0.514 / 0.417 |
| val วัตถุดิบดิบที่เพิ่ม (`results/baseline_n_v2_on_v2added/`) | 109 | 0.622 / 0.392 | **0.930 / 0.674** |

ราย class (mAP50-95, #1 → #2) · R = recall

| class | val วัตถุดิบดิบ (instance) | val เดิม v1 (instance) |
|---|---|---|
| pork | **0.114 → 0.424** (26) · R 0.077 → 0.614 | 0.143 → 0.133 (145) |
| shrimp | 0.402 → 0.604 (28) · R 0.363 → 0.964 | 0.180 → 0.227 (48) |
| ham | 0.383 → 0.600 (11) · R 0.000 → 1.000 | 0.205 → 0.147 (11) |
| bacon | 0.000 → 0.281 (18) · R 0.000 → 0.611 | 0.308 → 0.123 (7) |
| steak | 0.268 → 0.521 (33) | 0.382 → 0.366 (276) |
| chicken duck | 0.474 → 0.695 (56) | 0.341 → 0.310 (315) |

- val วัตถุดิบดิบ: ดีขึ้นทั้ง 27 class ที่มีใน val นี้ (เฉลี่ย +0.28) · confusion: เดิม pork ดิบถูกทายเป็น egg 4, bread 3, ไม่เจอ 3 ฯลฯ และ ham ถูกทายเป็น meat 7/11 → v2 เหลือ pork → bacon 2, steak 1, ไม่เจอ 1 · bacon → pork 4/18 (หมูดิบกับเบคอนดิบหน้าตาคล้ายกัน)
- val เดิม: เฉลี่ยราย class +0.009 (ขึ้น 38 ลง 34 จาก 73 class) → **ไม่ทำให้ของเดิมแย่ลง** · pork อาหารสุกบนจานยังแย่เท่าเดิม (ยังสับสนกับ steak 41/145 = 28% เดิม 32%) เพราะรูปที่เพิ่มเป็นของดิบ ไม่ได้ช่วย domain อาหารสุก
  - ลดลงมาก: sweet potato −0.448 (1 instance) · blueberries −0.193 (5) · bacon −0.185 (7) · cream cheese −0.110 (10) · ham −0.058 (11) → instance น้อยมาก ค่าแกว่งได้ อย่าตีความเกิน · bacon/ham ใน val เดิมสับสนกันเองมากขึ้น (ham → bacon 3/11)
  - ขึ้นมาก: flour +0.324 · squash +0.279 · lime +0.217 · cream +0.202 · avocado +0.191 · sugar +0.180 (flour, sugar ได้รูปเพิ่มจาก aicook)

**สรุป (บท 01: T, P, E)**: เพิ่ม E ที่ตรงกับ T (วัตถุดิบดิบในตู้เย็น) ทำให้ P บนงานจริงดีขึ้นมาก (val ดิบ mAP50-95 0.392 → 0.674) โดย P บนข้อมูลเดิมไม่ลดลง → ใช้ dataset v2 เป็นฐานของ Phase D ต่อ
- ⚠️ ข้อจำกัด: val ดิบมีแค่ 109 รูป และมาจากแหล่งเดียวกับ train ที่เพิ่ม (Roboflow ชุดเดียวกัน แม้กันรูปซ้ำต้นฉบับแล้ว) → ตัวเลขอาจสูงกว่าตอนใช้กับรูปจากตู้เย็นจริง
