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

| 0 | 2026-11-17 | YOLO11n detect | smoke test: `fraction=0.1` (1,006 รูป), batch 16, imgsz 640 | 1 | 0.0217 | 0.0171 | 0.161 | 0.037 | 2.2 นาที | 13.2 | – |

## รายละเอียดแต่ละการทดลอง

### #0 smoke (`runs/detect/smoke`)
- จุดประสงค์: ตรวจว่า pipeline ทำงาน **ไม่ใช่ผลที่ใช้เทียบ** (เทรนแค่ 1 epoch บน 10% ของ train)
- คำสั่ง: `scripts/train.py --name smoke --epochs 1 --fraction 0.1`
- ผล: สแกน label ได้ไม่มีรูป corrupt · label แบบ polygon แปลงเป็นกล่องเองได้ · ไม่มี warning · AMP ผ่าน
- GPU memory ตอนเทรน 2.49 GB ที่ batch 16 (จาก 6 GB) → baseline ใช้ batch 16 ได้
- ความเร็ว: 63 batch ใช้ 20.8 วินาที (~3.0 it/s) → ประมาณการ 1 epoch เต็ม (629 batch) ≈ 3.5 นาที + val ≈ 40 วินาที (**เป็นการประมาณ** ยังไม่ได้วัดจริง)
- loss epoch 1: train box 1.244 / cls 6.986 / dfl 1.527 · val box 1.161 / cls 5.255 / dfl 1.433
