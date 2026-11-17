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

## รายละเอียดแต่ละการทดลอง
