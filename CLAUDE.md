# smart-fridge-cv: งาน Computer Vision ของ Mini Project (Smart Fridge Recipe Recommender)

เจ้าของ: ปฏิมา ถิ่นสำราญ (T-Patima) · โรล **Computer Vision** · ใช้ **YOLO (Ultralytics YOLO11)**

## อ่านก่อนเริ่มทุกครั้ง
1. `PLAN.md`: แผนจนจบงาน (Phase A–G) + ตารางผูกเนื้อหาที่เรียนเข้ากับงาน CV + โครงคลิปนำเสนอ
2. `../mini-project-smart-fridge.md`: ภาพรวมโปรเจกต์ทั้งทีม + ผลตรวจ dataset + ประเด็นที่ต้องลงมติ
3. `data/split_report.md`: จำนวนรูปต่อ class ใน train/val/test
4. `experiments.md` (ถ้ามีแล้ว): ผลการทดลองที่ทำไปแล้ว

แล้วบอกผู้ใช้ว่าตอนนี้อยู่ Phase ไหน ทำอะไรเสร็จแล้ว และขั้นถัดไปคืออะไร ก่อนลงมือ

## ข้อมูล
- dataset: `data/foodseg_yolo/` (แตกจาก `../foodseg_fridge_thai_yolo.zip`, gitignore)
  - 75 class, label แบบ polygon (YOLO segmentation format) · train 10,055 / val 1,959 / test 1,317
  - config: `data/foodseg_yolo/data.yaml` (สำเนาที่ track ใน git: `data/data.yaml`) · ไม่มี `path:` ตั้งใจลบออก
  - test แบ่งจาก val ด้วย `scripts/make_test_split.py` (seed 42) ห้ามแบ่งใหม่ถ้าไม่ได้สั่ง
- ไม่มีข้าวและพาสต้าใน dataset · มติ 75 vs 12 class ยังรอทีม (ค่าเริ่มต้น 75)

## เครื่องและข้อจำกัด
- Windows 11 · GPU NVIDIA RTX 4050 Laptop 6 GB · Colab เป็นทางสำรอง
- Windows Application Control บล็อก exe ที่สร้างใหม่:
  - สร้าง venv ด้วย `C:\Users\patim\miniconda3\python.exe -m venv .venv` (ห้าม `uv venv`)
  - เรียกเครื่องมือผ่าน `python -m ...` ไม่ใช่ `.exe` (เช่น `python -m nbconvert`)
  - `DLL load failed ... Application Control` ครั้งแรก ให้รันซ้ำ
- Python ที่รัน `-I` จะไม่อ่าน `PYTHONIOENCODING` ใช้ `-X utf8` แทนเมื่อพิมพ์ภาษาไทย
- dataloader บน Windows: ใช้ `workers=2` หรือน้อยกว่า และสคริปต์ที่เทรนต้องมี `if __name__ == "__main__":`

## กติกา
- ตอบภาษาไทย ศัพท์เทคนิคใช้อังกฤษ · อ้างบทในสไลด์ (`../.claude/course-text/`) ส่วนเกินสไลด์ติดป้าย "เนื้อหาเสริม"
- งานนี้ต้องใช้ทำคลิปนำเสนอ: ทุกการทดลองบันทึกลง `experiments.md` พร้อมกราฟ และโยงกับบทที่เรียนตาม `PLAN.md` ข้อ 1
- สัมภาษณ์ผู้ใช้ (plan mode) ก่อนเขียนโค้ดงานใหม่ · commit ทีละขั้น · push/PR/merge เมื่อผู้ใช้สั่งเท่านั้น
- ห้ามแต่งตัวเลข · ใช้ test ครั้งเดียวตอนจบ
- git: repo ในเครื่อง (ยังไม่มี remote) · ชื่อผู้ commit: `git -c user.name="Patima Tinsumran" commit ...`
