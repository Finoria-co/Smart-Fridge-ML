# Person5 — Recipe Recommender (KNN)

ระบบแนะนำสูตรอาหารจากวัตถุดิบที่มีอยู่ โดยใช้ **K-Nearest Neighbors (KNN)** พร้อมคำนวณเปอร์เซ็นต์วัตถุดิบที่ตรงกัน

---

## ไฟล์ในโปรเจกต์

| ไฟล์ | คำอธิบาย |
|---|---|
| `model.ipynb` | Notebook สำหรับสร้างและเทรนโมเดล |
| `model.pkl` | ไฟล์โมเดลที่ถูก save ไว้พร้อมใช้งาน |
| `recipe_recommender.py` | Module สำหรับเรียกใช้งานจากระบบหลัก |
| `recipes_dataset/recipes_dataset.csv` | Dataset สูตรอาหาร 50 เมนู (ไทย 25 / ตะวันตก 25) |

---

## วิธีใช้งาน

```python
from person5.recipe_recommender import recommend

results = recommend(['หมูสับ', 'ไข่', 'กะเพรา'], top_n=5)

for r in results:
    print(r['name_th'], f"{r['match_pct']}%")
    print("  มีแล้ว:", r['have'])
    print("  ขาด:  ", r['missing'])
```

### ตัวอย่าง Output

```
ผัดกะเพราหมูสับไข่ดาว 100.0%
  มีแล้ว: ['หมู', 'ไข่', 'กะเพรา']
  ขาด:   []
```

---

## ค่าที่ส่งกลับ (`recommend()`)

| Field | ประเภท | คำอธิบาย |
|---|---|---|
| `recipe_id` | str | รหัสสูตรอาหาร เช่น `R001` |
| `name_th` | str | ชื่อภาษาไทย |
| `name_en` | str | ชื่อภาษาอังกฤษ |
| `cuisine` | str | ประเภทอาหาร (`Thai` / `Western`) |
| `match_pct` | float | เปอร์เซ็นต์วัตถุดิบที่มีครบ |
| `have` | list | วัตถุดิบที่มีแล้ว |
| `missing` | list | วัตถุดิบที่ยังขาด |
| `distance` | float | ระยะห่าง cosine (ยิ่งน้อยยิ่งใกล้) |

---

## วัตถุดิบที่รองรับ (18 ประเภท)

`หมู` · `ไก่` · `กุ้ง` · `เบคอน` · `ไข่` · `ข้าว` · `พาสต้า` · `ขนมปัง` · `กะเพรา` · `มะเขือเทศ` · `ชีส` · `ผักกาด` · `ปลา` · `เนื้อ` · `อาหารทะเล` · `เส้น` · `กะทิ` · `เห็ด`

---

## ผลการทดสอบ

| Metric | ผลลัพธ์ |
|---|---|
| Top-1 Accuracy | **90%** |
| Top-3 Accuracy | **100%** |
| Average Response Time | **2.75 ms** |

ทดสอบด้วย 10 test cases ครอบคลุมทั้งอาหารไทยและตะวันตก
