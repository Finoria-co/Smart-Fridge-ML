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
| 1 | 2026-11-17 | YOLO11n detect | **baseline** (Colab L4): batch 32, imgsz 640, patience 15, seed 0, cache ram | 50 (best 46) | **0.5101** | **0.4090** | 0.487 | 0.522 | 46.3 นาที | 1.3 (L4) | 01, 03, 05 |

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
- เวลา 46.3 นาทีตาม `results.csv` (train.py รายงาน 47.4 นาทีรวมโหลดข้อมูล) ≈ 1.5 CU
- **ทำซ้ำได้ (reproducible)**: เซลล์ 5 ถูกรันจนจบ 2 รอบ (`baseline_n`, `baseline_n-3` ใน Drive) ได้ P/R/mAP/loss ตรงกันทุกคอลัมน์ทั้ง 50 epoch เพราะ `seed=0` + `deterministic=True` · `baseline_n-2` หยุดหลัง epoch 1 (ไม่ใช้)
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
