# 🧊 Smart Fridge Recipe Recommender
> **วิชา:** Machine Learning Mini-Project (ชั้นปีที่ 3)  
> **ระบบตรวจจับภาพถ่ายวัตถุดิบในตู้เย็นและวิเคราะห์แนะนำเมนูอาหารไทย-ฝรั่งด้วย Machine Learning**

---

## 📌 บทนำและวัตถุประสงค์ (Overview)
ปัญหาของสดเหลือในตู้เย็นจนหมดอายุ (Food Waste) และความลังเลในการทำอาหาร ("เย็นนี้กินอะไรดี?") เป็นปัญหาที่พบบ่อยในชีวิตประจำวัน  
โปรเจกต์นี้จึงผสานเทคโนโลยี **Computer Vision (YOLO11s)** เพื่อตรวจจับของสดจากรูปถ่าย เข้ากับ **Thai NLP** และ **Recommender System (Weighted Cosine Similarity)** เพื่อแนะนำเมนูอาหารที่เหมาะสมที่สุดจากวัตถุดิบที่มีจริง พร้อมคำนวณวัตถุดิบที่ยังขาดและแสดงขั้นตอนการปรุงทีละสเต็ป

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)
ระบบทำงานแบบแยกโมดูลอิสระ (Modular Design) ผ่าน **4-Stage Pipeline**:
1. **Input Layer:** รับรูปถ่ายตู้เย็น (Drag & Drop) หรือข้อความภาษาไทย
2. **Detection & Extraction Layer:**
   - *รูปภาพ:* ประมวลผลด้วยโมเดล **YOLO11s** เพื่อระบุตำแหน่ง (Bounding Box) และประเภทวัตถุดิบ
   - *ข้อความ:* ทำ Preprocessing ตัดคำ และแมปคำพ้อง (Synonyms) เช่น หมูบด -> หมูสับ
3. **Core ML Recommender Layer:**
   - คำนวณความคล้ายคลึงด้วย **Weighted Cosine Similarity**
   - คำนวณสัดส่วนความครอบคลุมของวัตถุดิบจริง (True Ingredient Coverage)
   - กรองเครื่องปรุงพื้นฐานประจำครัว (Pantry Staples) ออกจากรายการของที่ขาด
4. **Output Layer:**
   - จัดอันดับเมนูที่ทำได้ทันที (Ready to cook) และเมนูที่ขาดของน้อยที่สุดขึ้นก่อน
   - แสดงคะแนน Match %, แคลอรี, เวลาปรุง, ระดับความเผ็ด และวิธีปรุงทีละขั้นตอน

---

## 👥 สมาชิกกลุ่มและหน้าที่รับผิดชอบ (Team Roles)
| ลำดับ | บทบาทหน้าที่ | หน้าที่หลัก |
| :---: | :--- | :--- |
| **คนที่ 1** | Image Data Engineer | รวบรวมชุดภาพถ่ายวัตถุดิบ, ทำ Labeling & Augmentation บน Roboflow |
| **คนที่ 2** | Computer Vision ML Lead | ฝึกสอนและประเมินผลโมเดล YOLO11s ตรวจจับวัตถุดิบ (`models/best.pt`) |
| **คนที่ 3** | Recipe Data Engineer | ออกแบบและดูแลฐานข้อมูล 50 สูตรอาหารไทย-ฝรั่ง (`data/recipes_dataset.csv`) |
| **คนที่ 4** | Text Processing & NLP | พัฒนาระบบสกัดคำพ้องวัตถุดิบภาษาไทย (`modules/text_extractor.py`) |
| **คนที่ 5** | Core Recipe ML Lead | ออกแบบอัลกอริทึมจับคู่เมนู Weighted Cosine & Coverage (`modules/recipe_recommender.py`) |
| **คนที่ 6** | Web UI & Full-Stack Lead | พัฒนาระบบรวมศูนย์ Flask Web API, Frontend UI, สไลด์ และ Live Demo |

---

## 🚀 วิธีการติดตั้งและรันโปรเจกต์ (Quick Start)

### 1. ติดตั้ง Dependencies
```powershell
# สร้าง Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# ติดตั้งแพ็กเกจที่จำเป็น
pip install flask pillow ultralytics
```

### 2. รันแอปพลิเคชัน
```powershell
python app.py
```
เปิดเบราว์เซอร์ไปที่: **`http://127.0.0.1:5000`**

### 3. รัน Integration Test
```powershell
python test_app.py
```
