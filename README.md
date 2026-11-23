#  Smart Fridge Recipe Recommender
> **วิชา:** Machine Learning Mini-Project (ชั้นปีที่ 3)  
> **ระบบตรวจจับภาพถ่ายวัตถุดิบในตู้เย็นและวิเคราะห์แนะนำเมนูอาหารไทย-ฝรั่งด้วย Machine Learning (80 เมนู • 4 โมเดล ML)**  
>  **Live Web Application (GitHub Pages):** [https://finoria-co.github.io/Smart-Fridge-ML/](https://finoria-co.github.io/Smart-Fridge-ML/)

---

##  บทนำและวัตถุประสงค์ (Overview)
ปัญหาของสดเหลือในตู้เย็นจนหมดอายุ (Food Waste) และความลังเลในการทำอาหาร ("เย็นนี้กินอะไรดี?") เป็นปัญหาที่พบบ่อยในชีวิตประจำวัน  
โปรเจกต์นี้จึงผสานเทคโนโลยี **Computer Vision (YOLO11s / WebAssembly ONNX)** เพื่อตรวจจับของสดจากรูปถ่าย เข้ากับ **Recommender System (Weighted Cosine Similarity, KNN, TF-IDF และ Random Forest)** เพื่อแนะนำเมนูอาหารที่เหมาะสมที่สุดจากวัตถุดิบที่มีจริง พร้อมคำนวณวัตถุดิบที่ยังขาดและแสดงขั้นตอนการปรุงทีละสเต็ป

---

##  สถาปัตยกรรมระบบ (System Architecture)
ระบบทำงานแบบแยกโมดูลอิสระ (Modular Design) ผ่าน **4-Stage Pipeline**:
1. **Input Layer (Computer Vision):** รับรูปถ่ายตู้เย็นและวัตถุดิบ (Upload / Drag & Drop จากกล้องหรือไฟล์)
2. **Detection Layer:**
   - ประมวลผลภาพด้วยโมเดล **YOLO11s** (รันผ่าน Python Backend หรือ In-Browser ONNX Runtime) เพื่อระบุตำแหน่ง (Bounding Box) และประเภทวัตถุดิบพร้อมค่าความมั่นใจ (Confidence Score)
   - แมปคลาสวัตถุดิบเข้าสู่มาตรฐานของระบบ (Standard Ingredients)
3. **Core ML Recommender Layer:**
   - คำนวณความคล้ายคลึงด้วย **Weighted Cosine Similarity** ร่วมกับโมเดลเปรียบเทียบ **KNN**, **TF-IDF Vectorizer** และ **Random Forest**
   - คำนวณสัดส่วนความครอบคลุมของวัตถุดิบจริง (True Ingredient Coverage)
   - กรองเครื่องปรุงพื้นฐานประจำครัว (Pantry Staples) ออกจากรายการของที่ขาด
4. **Output Layer:**
   - จัดอันดับเมนูที่ทำได้ทันที (Ready to cook) และเมนูที่ขาดของน้อยที่สุดขึ้นก่อนอย่างเคร่งครัด
   - แสดงคะแนน Match %, แคลอรี, เวลาปรุง, ระดับความเผ็ด และวิธีปรุงทีละขั้นตอน

---

##  สมาชิกกลุ่มและหน้าที่รับผิดชอบ (Team Roles)

| ลำดับ | รหัสนักศึกษา | ชื่อ - นามสกุล | บทบาทหน้าที่ | หน้าที่รับผิดชอบหลัก |
| :---: | :---: | :------- | :--- | :--- |
| **1** | 6752300241 | **อัมพร ชินพร** | Image Data Engineer | รวบรวมชุดภาพถ่ายวัตถุดิบ, ทำ Labeling & Augmentation บน Roboflow |
| **2** | 6752300461 | **ปฏิมา ถิ่นสำราญ** | Computer Vision ML Lead | ฝึกสอนและประเมินผลโมเดล YOLO11s ตรวจจับวัตถุดิบ (`models/best.pt`) |
| **3** | 6752301123 | **นัทธนัท นุ่มนวล** | Recipe Data Engineer | ออกแบบและดูแลฐานข้อมูล 80 สูตรอาหารไทย-ฝรั่ง (`data/recipes_dataset.csv`) |
| **4** | 6752300739 | **รุจิรัตน์ แสงชัย** | Text Processing & NLP | พัฒนาระบบสกัดคำพ้องวัตถุดิบภาษาไทย (`modules/text_extractor.py`) |
| **5** | 6752301239<br>6752301085 | **ขวัญมนัส คล้ายมี**<br>**กัลยรัตน์ โพธิสาร** | Core Recipe ML Lead | ออกแบบอัลกอริทึมจับคู่เมนูและเปรียบเทียบโมเดล (Weighted Cosine, KNN, TF-IDF, Random Forest) |
| **6** | 6752301085 | **กัลยรัตน์ โพธิสาร** | Web UI & Full-Stack Lead | พัฒนาระบบรวมศูนย์ Flask Web API, Frontend UI (In-Browser ONNX), สไลด์ และ GitHub Pages |

---

##  วิธีการติดตั้งและรันโปรเจกต์ (Quick Start)

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
