# 🧊 คู่มือการใช้งานระบบ Smart Fridge ML (สำหรับคนที่ 6 และทีมงาน)

โปรเจกต์นี้พัฒนาขึ้นมาเพื่อทำหน้าที่เป็น **ศูนย์กลางการเชื่อมระบบ (Full Integration & Web UI)** ของกลุ่ม 6 คน โดยมีสถาปัตยกรรมแบบแยกโมดูลอิสระ (Modular Architecture) ทำให้คนที่ 6 สามารถรันเทสระบบและนำเสนอสดได้ทันที แม้เพื่อนคนอื่นจะยังเทรนโมเดลไม่เสร็จ

---

## 🚀 1. วิธีเปิดรันเซิร์ฟเวอร์เว็บแอปพลิเคชัน (How to Run)

เปิด PowerShell หรือ Terminal ในโฟลเดอร์โปรเจกต์นี้ แล้วรันคำสั่ง:

```powershell
# รันเปิดเว็บเซิร์ฟเวอร์ Flask
.\.venv\Scripts\python.exe app.py
```

เมื่อขึ้นข้อความ:
```
* Running on http://127.0.0.1:5000
```
ให้เปิดเว็บเบราว์เซอร์ไปที่: **`http://127.0.0.1:5000`**

---

## 🔌 2. วิธีการนำโค้ดและโมเดลจริงของเพื่อนมาเสียบ (Plug-and-Play)

โครงสร้างโฟลเดอร์ถูกแยกตามบทบาทของเพื่อนแต่ละคนไว้อย่างชัดเจน:

```
miniproj/
├── app.py                         # [คนที่ 6] Backend API เชื่อมทุกส่วน
├── templates/
│   └── index.html                 # [คนที่ 6] Frontend UI สวยงาม รองรับ Drag & Drop และ Demo
├── modules/
│   ├── vision_detector.py         # [คนที่ 2] ช่องรับโมเดลตรวจจับรูปภาพ (YOLOv8)
│   ├── text_extractor.py          # [คนที่ 4] ตัวตัดคำและ Synonym Mapping
│   └── recipe_recommender.py      # [คนที่ 5] โมเดล ML แนะนำเมนู (Weighted Cosine / KNN)
├── data/
│   └── recipes_dataset.csv        # [คนที่ 3] ฐานข้อมูลสูตรอาหาร 12 ฟีเจอร์
├── static/
│   ├── demo_samples/              # [คนที่ 6] ภาพจำลองสำหรับใช้กด Live Demo บนเวที
│   └── uploads/                   # ที่เก็บภาพที่อัปโหลดและภาพที่วาดกรอบ Bounding Box แล้ว
├── presentation/
│   └── presentation_slides.md     # [คนที่ 6] สคริปต์สไลด์รวม 10 หน้า พร้อมบทพูด 6 คน
└── demo/
    └── live_demo_script.md        # [คนที่ 6] สคริปต์ขั้นตอนการกด Demo โชว์อาจารย์
```

### เมื่อเพื่อนส่งงานมา ให้ทำดังนี้:
1. **เมื่อคนที่ 2 ส่งโมเดล YOLOv8 มา:**
   * นำไฟล์โมเดลที่เทรนเสร็จมาตั้งชื่อว่า `best.pt` แล้ววางไว้ในโฟลเดอร์ `models/best.pt`
   * ระบบใน `modules/vision_detector.py` จะตรวจพบและสลับไปรันโมเดลจริงให้อัตโนมัติทันที
2. **เมื่อคนที่ 3 อัปเดตเมนูเพิ่ม:**
   * นำไฟล์ CSV มาวางทับที่ `data/recipes_dataset.csv` ได้ทันที (ระบบรองรับเมนูใหม่โดยไม่ต้องแก้โค้ด)
3. **เมื่อคนที่ 4 เพิ่มคำพ้อง (Synonyms):**
   * เพิ่มคำศัพท์ลงใน Dictionary `SYNONYM_DICT` ในไฟล์ `modules/text_extractor.py`
4. **เมื่อคนที่ 5 ปรับจูนน้ำหนักโมเดล:**
   * ปรับแต่งค่าใน `modules/recipe_recommender.py`

---

## 📊 3. สิ่งที่คนที่ 6 ต้องเตรียมส่งและนำเสนอ

1. **ระบบเว็บแอปพลิเคชันที่ทำงานได้สมบูรณ์:**
   * [app.py](file:///C:/Users/Win11%20Pro/Documents/MIX/PIM/pim/pim/3-ปีสาม/ML/miniproj/app.py)
   * [templates/index.html](file:///C:/Users/Win11%20Pro/Documents/MIX/PIM/pim/pim/3-ปีสาม/ML/miniproj/templates/index.html)
2. **ไฟล์สไลด์นำเสนอรวม 10 สไลด์ (ก๊อปปี้ลง Canva หรือ Google Slides ได้ทันที):**
   * [presentation/presentation_slides.md](file:///C:/Users/Win11%20Pro/Documents/MIX/PIM/pim/pim/3-ปีสาม/ML/miniproj/presentation/presentation_slides.md)
3. **บทพูดสำหรับ Live Demo บนเวทีทีละคลิก:**
   * [demo/live_demo_script.md](file:///C:/Users/Win11%20Pro/Documents/MIX/PIM/pim/pim/3-ปีสาม/ML/miniproj/demo/live_demo_script.md)
