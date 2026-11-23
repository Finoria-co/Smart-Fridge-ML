# Split report: dataset v2 (v1 + วัตถุดิบดิบจาก Roboflow)

สร้างด้วย `scripts/build_dataset_v2.py` · seed 42 · รูปใหม่แบ่ง 70/15/15 · v1 ไม่เปลี่ยน · CAP 300 รูป/class เป้าหมาย
ที่มา: `data/sources.md` · map class: `data/rf_class_map.yaml`

| split | v1 | เพิ่ม | v2 |
|---|---|---|---|
| train | 10,055 | 490 | 10,545 |
| val | 1,959 | 109 | 2,068 |
| test | 1,317 | 110 | 1,427 |

## การคัดรูปจาก Roboflow

- aicook-lcv4d: ซ้ำกับ v1: 73
- aicook-lcv4d: ตัดเพราะมี class DROP: 78
- aicook-lcv4d: เลือก: 320
- aicook-lcv4d: ไฟล์ทั้งหมด: 516
- aicook-lcv4d: ไม่มี class เป้าหมาย: 45
- food-ingredients-x2: ซ้ำต้นฉบับเดิม (augment): 1,107
- food-ingredients-x2: ตัดเพราะมี class DROP: 166
- food-ingredients-x2: เลือก: 270
- food-ingredients-x2: ไฟล์ทั้งหมด: 1,742
- food-ingredients-x2: ไม่มี class เป้าหมาย: 199
- fridge-objects: ซ้ำกับ v1: 3
- fridge-objects: เลือก: 90
- fridge-objects: ไฟล์ทั้งหมด: 850
- fridge-objects: ไม่มี class เป้าหมาย: 757
- meat-data-sets: ซ้ำต้นฉบับเดิม (augment): 104
- meat-data-sets: ตัดเพราะมี class DROP: 5
- meat-data-sets: เลือก: 29
- meat-data-sets: ไฟล์ทั้งหมด: 156
- meat-data-sets: ไม่มี class เป้าหมาย: 18
- รวมหลัง CAP: 709

## รูปต่อ class ที่เปลี่ยน (v1 → เพิ่ม) แยก train / val / test

| id | class | train v1 | +train | val v1 | +val | test v1 | +test |
|---|---|---|---|---|---|---|---|
| 0 | cheese butter | 604 | +169 | 161 | +40 | 107 | +42 |
| 1 | milk | 377 | +84 | 102 | +18 | 68 | +18 |
| 2 | egg | 657 | +59 | 151 | +13 | 100 | +13 |
| 3 | apple | 303 | +68 | 78 | +14 | 52 | +14 |
| 5 | banana | 244 | +103 | 73 | +25 | 49 | +29 |
| 6 | strawberry | 463 | +130 | 119 | +28 | 80 | +39 |
| 11 | steak | 749 | +133 | 209 | +29 | 139 | +29 |
| 12 | pork | 474 | +119 | 117 | +26 | 78 | +26 |
| 13 | chicken duck | 913 | +203 | 254 | +44 | 170 | +43 |
| 17 | shrimp | 116 | +129 | 31 | +28 | 21 | +33 |
| 18 | bread | 1074 | +44 | 275 | +10 | 184 | +10 |
| 19 | corn | 360 | +107 | 94 | +23 | 63 | +28 |
| 22 | potato | 830 | +118 | 199 | +31 | 133 | +26 |
| 25 | tomato | 1048 | +143 | 285 | +31 | 190 | +39 |
| 34 | carrot | 1052 | +100 | 284 | +16 | 189 | +20 |
| 42 | onion | 429 | +97 | 107 | +15 | 71 | +36 |
| 43 | pepper | 430 | +2 | 115 | +0 | 77 | +0 |
| 44 | green beans | 194 | +100 | 51 | +21 | 34 | +26 |
| 46 | mushroom | 263 | +93 | 69 | +21 | 46 | +19 |
| 48 | blueberries | 37 | +106 | 5 | +22 | 4 | +23 |
| 57 | spinach | 40 | +97 | 7 | +23 | 5 | +30 |
| 62 | bacon | 18 | +63 | 4 | +14 | 2 | +13 |
| 63 | cream | 30 | +86 | 7 | +23 | 5 | +21 |
| 65 | flour | 16 | +105 | 6 | +26 | 4 | +23 |
| 66 | ham | 32 | +55 | 7 | +11 | 5 | +12 |
| 67 | lime | 40 | +80 | 8 | +19 | 5 | +18 |
| 69 | sugar | 17 | +114 | 4 | +25 | 3 | +24 |
| 70 | sweet potato | 16 | +94 | 1 | +20 | 0 | +25 |
