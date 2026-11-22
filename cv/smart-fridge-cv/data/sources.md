# ที่มาของข้อมูล

## dataset หลัก (v1)
`foodseg_fridge_thai_yolo` (เพื่อนในทีมรวมและคลีน): FoodSeg103 + รูปตู้เย็นจาก Roboflow + `5-thai-herbs` · 75 class · polygon
รายละเอียด: `../mini-project-smart-fridge.md` ข้อ 3

## รูปวัตถุดิบดิบที่เพิ่มใน v2 (Roboflow Universe, เลือก 2026-11-17)
เหตุผล: รูป pork ใน v1 เป็นอาหารปรุงสุกบนจาน ไม่ตรงกับงานจริง (วัตถุดิบดิบในตู้เย็น) → `experiments.md` #1 · บท 01: *"Is training experience representative of performance goal?"*

| ชุด | version | license | ใช้ทำอะไร | ลิงก์ |
|---|---|---|---|---|
| aicook (Karel Cornelis) | v1 (ไม่มี augmentation) | MIT | shrimp, ham ในตู้เย็นจริง | https://universe.roboflow.com/karel-cornelis-q2qqg/aicook-lcv4d |
| Fridge objects | v9 (ไม่มี augmentation) | CC BY 4.0 | bacon ในตู้เย็น | https://universe.roboflow.com/fooddetection-essdj/fridge-objects |
| food-ingredients-x2 | v1 | CC BY 4.0 | pork ดิบ (แพ็ก/กระทะ) | https://universe.roboflow.com/yolov5-rznzw/food-ingredients-x2 |
| Meat Data Sets | v4 | CC BY 4.0 | pork, shrimp ดิบ (ภาพ stock) | https://universe.roboflow.com/thesis-6tgwg/meat-data-sets |

ไม่ใช้: `fridgeitems/object-fridge-items` (มี aicook ทั้ง 516 รูป และซ้ำกับรูปตู้เย็นใน v1 507 รูป)

การแปลงชื่อ class: `data/rf_class_map.yaml` · สร้าง dataset: `scripts/build_dataset_v2.py`
CC BY 4.0 ต้องอ้างอิงที่มา → ใส่ตารางนี้ในสไลด์ Method / เครดิตท้ายคลิป
