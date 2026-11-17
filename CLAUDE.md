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
- Windows 11 · GPU NVIDIA RTX 4050 Laptop 6 GB
- **เทรนหลักบน Colab** (ผู้ใช้เลือก 2026-11-17 เพื่อไม่ให้เครื่องหนัก): `notebooks/colab_train.ipynb` · L4 GPU กิน ~1.94 CU/ชม. · บัญชีไม่ได้ subscribe ใช้ CU ที่ซื้อเพิ่ม (100 CU / $9.99) → **ประหยัด CU**: ตัด runtime ทันทีเมื่อเทรนเสร็จ
  - Drive: `MyDrive/Mini_Project_Finoria_ML/` (ในเครื่องอ่านตรงได้ที่ `G:\My Drive\Mini_Project_Finoria_ML` ผ่าน Google Drive for desktop แบบ Stream · Git Bash: `/g/My Drive/...` · ตัวเชื่อม google-drive ของ plugin ล็อกอินไม่ผ่าน ไม่ต้องลอง) มี `foodseg_yolo_split.zip` (สร้างด้วย `scripts/zip_dataset.py`, ไฟล์ zip อยู่ที่ `data/` ในเครื่อง) + `train.py` (ถ้าแก้ `scripts/train.py` ต้องอัปโหลดทับ) · ผลเทรนอยู่ที่ `MyDrive/Mini_Project_Finoria_ML/runs/detect/<name>`
  - การทดลองบน Colab ใช้ batch 32, workers 8, cache ram → เทียบกันเฉพาะกับ run บน Colab
- **งานที่ใช้ PyTorch ในเครื่อง (สำรอง/งานสั้น) ใช้ WSL2 Ubuntu** (ผู้ใช้เลือก 2026-11-17): Smart App Control บล็อก `torch\lib\torch.dll` ถาวรบน Windows (ลองทั้ง torch 2.11 และ 2.9.1 รันซ้ำแล้วก็ไม่ผ่าน)
  - venv: `/root/venvs/smart-fridge` (Python 3.12, torch 2.11.0+cu128, ultralytics 8.4.215) · ติดตั้งใหม่ด้วย `scripts/setup_wsl.sh`
  - dataset สำเนาบน ext4: `/root/data/foodseg_yolo/data.yaml` (อ่านเร็วกว่า `/mnt/c`) · โค้ดและ git อยู่โฟลเดอร์นี้ (`/mnt/c/Users/patim/Project/Finoria/smart-fridge-cv`)
  - เรียกจาก PowerShell: `wsl.exe -d Ubuntu --cd /mnt/c/Users/patim/Project/Finoria/smart-fridge-cv -- /root/venvs/smart-fridge/bin/python scripts/xxx.py` (quoting ซับซ้อนให้เขียนเป็นไฟล์สคริปต์ · จาก Git Bash ต้องใส่ `MSYS_NO_PATHCONV=1`)
  - RAM ใน WSL มีราว 7 GB: `workers` ≤ 4, ห้าม `cache="ram"`
  - `.venv/` บน Windows มี torch ที่โหลดไม่ได้ ใช้ได้เฉพาะงานที่ไม่ใช้ torch
- Windows Application Control บล็อก exe ที่สร้างใหม่:
  - สร้าง venv ด้วย `C:\Users\patim\miniconda3\python.exe -m venv .venv` (ห้าม `uv venv`)
  - เรียกเครื่องมือผ่าน `python -m ...` ไม่ใช่ `.exe` (เช่น `python -m nbconvert`)
  - `DLL load failed ... Application Control` ครั้งแรก ให้รันซ้ำ
- Python ที่รัน `-I` จะไม่อ่าน `PYTHONIOENCODING` ใช้ `-X utf8` แทนเมื่อพิมพ์ภาษาไทย
- dataloader บน Windows: ใช้ `workers=2` หรือน้อยกว่า และสคริปต์ที่เทรนต้องมี `if __name__ == "__main__":`

## กติกา
- ตอบภาษาไทย ศัพท์เทคนิคใช้อังกฤษ · อ้างบทในสไลด์ (`../.claude/course-text/`) ส่วนเกินสไลด์ติดป้าย "เนื้อหาเสริม"
- งานนี้ต้องใช้ทำคลิปนำเสนอ: ทุกการทดลองบันทึกลง `experiments.md` พร้อมกราฟ และโยงกับบทที่เรียนตาม `PLAN.md` ข้อ 1
- สัมภาษณ์ผู้ใช้ (plan mode) ก่อนเขียนโค้ดงานใหม่ · ห้ามแต่งตัวเลข · ใช้ test ครั้งเดียวตอนจบ
- **git (ผู้ใช้สั่ง 2026-11-17)**: remote = https://github.com/Finoria-co/smart-fridge-cv (public, branch `main`)
  - **ทุกครั้งที่แก้หรืออัปเดตไฟล์ใน repo ต้อง commit เก็บไว้เสมอ** (ไม่ปล่อย working tree ค้าง) · commit แยกตามขั้น/เฟส (ข้อความขึ้นต้นด้วย Phase หรือหมวด) เพื่อ `git revert` ทีละขั้นได้ · **push ขึ้น main ได้ทันทีหลังทุก commit ไม่ต้องถาม**
  - ชื่อผู้ commit: `git -c user.name="Patima Tinsumran" commit ...` · ห้ามใส่รหัสนักศึกษาหรือข้อมูลส่วนตัวในไฟล์ (repo เป็น public)
  - PR/merge/force-push/ลบ branch ยังต้องให้ผู้ใช้สั่ง
- ผลเทรน: copy run จาก Drive มาไว้ `runs/detect/<name>/` แล้วรัน (ใน WSL) `scripts/analyze_run.py runs/detect/<name> <name>` (copy csv + กราฟหลัก + กราฟ train vs val loss ไป `results/<name>/`) และ `scripts/eval_per_class.py runs/detect/<name>/weights/best.pt <name>` (mAP ราย class บน val) แล้ว commit · `runs/` และ `*.pt` ไม่เข้า git (weights อยู่ที่ Drive, best.pt ตัวสุดท้ายแนบเป็น GitHub Release)
