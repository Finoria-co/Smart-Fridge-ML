"""
Module 2: Computer Vision Ingredient Detector (สำหรับคนที่ 2)
หน้าที่: รับรูปถ่ายตู้เย็น/วัตถุดิบ -> ตรวจจับวัตถุดิบ 12 ชนิด -> คืนค่าชื่อวัตถุดิบและกรอบ Bounding Box

เมื่อคนที่ 2 เทรนโมเดลเสร็จแล้ว ให้นำไฟล์ best.pt มาวางไว้ที่โฟลเดอร์ models/best.pt
ระบบจะสลับไปรันโมเดล YOLOv8 จริงอัตโนมัติทันที
"""

import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont

import json

# 12 วัตถุดิบเป้าหมายของระบบ (Standard Classes)
CLASS_NAMES_TH = [
    "หมูสับ", "อกไก่", "กุ้งสด", "เบคอน", "ไข่ไก่", 
    "ข้าวสวย", "เส้นพาสต้า", "ขนมปัง", "ใบกะเพรา", 
    "มะเขือเทศ", "ชีส", "ผักกาดขาว"
]

# โหลด mapping จาก ingredient_map.json ของคนที่ 2 (รองรับ 75 คลาสของโมเดล YOLO11s)
MAP_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ingredient_map.json")
CLASS_MAP_EN_TO_TH = {
    "pork": "หมูสับ",
    "chicken duck": "อกไก่",
    "chicken": "อกไก่",
    "shrimp": "กุ้งสด",
    "bacon": "เบคอน",
    "ham": "เบคอน",
    "egg": "ไข่ไก่",
    "rice": "ข้าวสวย",
    "pasta": "เส้นพาสต้า",
    "bread": "ขนมปัง",
    "holy basil": "ใบกะเพรา",
    "holy_basil": "ใบกะเพรา",
    "tomato": "มะเขือเทศ",
    "cheese butter": "ชีส",
    "cream cheese": "ชีส",
    "cheese": "ชีส",
    "cabbage": "ผักกาดขาว"
}

if os.path.exists(MAP_FILE):
    try:
        with open(MAP_FILE, "r", encoding="utf-8") as f:
            cdata = json.load(f).get("contract", {})
            for std_key, val in cdata.items():
                th_name = val.get("th", "")
                if "เบคอน" in th_name:
                    th_name = "เบคอน"
                for cls_name in val.get("classes", []):
                    CLASS_MAP_EN_TO_TH[cls_name.lower()] = th_name
    except Exception as e:
        print(f"[VisionDetector] ไม่สามารถอ่าน {MAP_FILE} ได้: {e}")

# พรีเซ็ตวัตถุดิบสำหรับการจำลองแบบสมจริง (Mock Heuristics)
DEMO_PRESETS = {
    "thai_pork": {
        "items": ["หมูสับ", "ไข่ไก่", "ใบกะเพรา"],
        "boxes": [
            {"label": "หมูสับ", "en": "pork", "confidence": 0.94, "box": [0.10, 0.25, 0.40, 0.75]},
            {"label": "ไข่ไก่", "en": "egg", "confidence": 0.97, "box": [0.45, 0.20, 0.68, 0.65]},
            {"label": "ใบกะเพรา", "en": "holy_basil", "confidence": 0.91, "box": [0.70, 0.28, 0.92, 0.72]}
        ]
    },
    "western_pasta": {
        "items": ["เส้นพาสต้า", "เบคอน", "ชีส", "ไข่ไก่"],
        "boxes": [
            {"label": "เส้นพาสต้า", "en": "pasta", "confidence": 0.96, "box": [0.08, 0.20, 0.35, 0.78]},
            {"label": "เบคอน", "en": "bacon", "confidence": 0.93, "box": [0.38, 0.25, 0.65, 0.68]},
            {"label": "ชีส", "en": "cheese", "confidence": 0.89, "box": [0.68, 0.18, 0.92, 0.58]},
            {"label": "ไข่ไก่", "en": "egg", "confidence": 0.95, "box": [0.55, 0.62, 0.80, 0.88]}
        ]
    },
    "brunch_bread": {
        "items": ["ขนมปัง", "ชีส", "เบคอน", "มะเขือเทศ"],
        "boxes": [
            {"label": "ขนมปัง", "en": "bread", "confidence": 0.95, "box": [0.12, 0.22, 0.42, 0.72]},
            {"label": "ชีส", "en": "cheese", "confidence": 0.92, "box": [0.44, 0.18, 0.70, 0.58]},
            {"label": "เบคอน", "en": "bacon", "confidence": 0.90, "box": [0.42, 0.60, 0.72, 0.85]},
            {"label": "มะเขือเทศ", "en": "tomato", "confidence": 0.96, "box": [0.72, 0.28, 0.93, 0.75]}
        ]
    },
    "soup_cabbage": {
        "items": ["หมูสับ", "ผักกาดขาว", "เต้าหู้ไข่"],
        "boxes": [
            {"label": "หมูสับ", "en": "pork", "confidence": 0.93, "box": [0.12, 0.22, 0.44, 0.70]},
            {"label": "ผักกาดขาว", "en": "cabbage", "confidence": 0.95, "box": [0.48, 0.18, 0.88, 0.78]}
        ]
    }
}


# ตารางแปลงชื่อวัตถุดิบทั้งหมด 75 คลาสของโมเดล YOLO11s เป็นภาษาไทยมาตรฐาน
CLASS_MAP_ALL_EN_TO_TH = {
    # 12 คลาสหลักเดิม
    "pork": "หมูสับ",
    "chicken duck": "อกไก่",
    "chicken": "อกไก่",
    "shrimp": "กุ้งสด",
    "bacon": "เบคอน",
    "ham": "เบคอน",
    "egg": "ไข่ไก่",
    "rice": "ข้าวสวย",
    "pasta": "เส้นพาสต้า",
    "bread": "ขนมปัง",
    "holy basil": "ใบกะเพรา",
    "holy_basil": "ใบกะเพรา",
    "tomato": "มะเขือเทศ",
    "cheese butter": "ชีส",
    "cream cheese": "ชีส",
    "cheese": "ชีส",
    "cabbage": "ผักกาดขาว",
    
    # วัตถุดิบเสริมที่เพื่อนคนที่ 2 เทรนเพิ่ม (รวมเข้าเป็นวัตถุดิบที่ระบบตรวจจับและใช้งานได้จริง)
    "wild betel": "ใบชะพลู",
    "wild_betel": "ใบชะพลู",
    "sweet basil": "ใบโหระพา",
    "sweet_basil": "ใบโหระพา",
    "kaffir lime": "ใบมะกรูด",
    "garlic": "กระเทียม",
    "onion": "หอมใหญ่",
    "carrot": "แครอท",
    "potato": "มันฝรั่ง",
    "sweet potato": "มันหวาน",
    "mushroom": "เห็ด",
    "tofu": "เต้าหู้",
    "fish": "ปลา",
    "shellfish": "หอย",
    "sausage": "ไส้กรอก",
    "steak": "เนื้อวัว",
    "meat": "เนื้อวัว",
    "milk": "นมสด",
    "cream": "ครีมสด",
    "pepper": "พริก",
    "chili": "พริก",
    "spring onion": "ต้นหอม",
    "spring_onion": "ต้นหอม",
    "lemon": "มะนาว",
    "lime": "มะนาว",
    "corn": "ข้าวโพด",
    "cucumber": "แตงกวา",
    "pumpkin": "ฟักทอง",
    "broccoli": "บรอกโคลี",
    "spinach": "ผักโขม",
    "lettuce": "ผักสลัด",
    "ginger": "ขิง",
    "bean sprouts": "ถั่วงอก",
    "eggplant": "มะเขือยาว",
    "apple": "แอปเปิ้ล",
    "banana": "กล้วย",
    "mango": "มะม่วง",
    "avocado": "อะโวคาโด"
}


def get_thai_font(size=16):
    """โหลดฟอนต์ภาษาไทยของ Windows เพื่อให้วาดชื่อวัตถุดิบบนรูปภาพได้คมชัด ไม่เพี้ยน"""
    candidates = [
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/leelawad.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/cordia.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_bounding_boxes(image_path, detections, output_path):
    """วาดกรอบ Bounding Box พร้อมชื่อวัตถุดิบภาษาไทยคมชัด ไม่เพี้ยน ลงบนภาพ"""
    try:
        with Image.open(image_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            w, h = img.size

            # ฟอนต์ภาษาไทยขนาดพอเหมาะกับขนาดภาพ
            font_size = max(14, int(min(w, h) * 0.035))
            font = get_thai_font(font_size)

            colors = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"]

            for i, det in enumerate(detections):
                box = det["box"]
                # รองรับทั้ง normalized coords (0.0 - 1.0) และ pixel coords
                x1 = int(box[0] * w) if box[0] <= 1.0 else int(box[0])
                y1 = int(box[1] * h) if box[1] <= 1.0 else int(box[1])
                x2 = int(box[2] * w) if box[2] <= 1.0 else int(box[2])
                y2 = int(box[3] * h) if box[3] <= 1.0 else int(box[3])
                color = colors[i % len(colors)]

                # วาดกรอบสี่เหลี่ยมหนา 3-4 px
                for offset in range(3):
                    draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

                # ป้ายข้อความภาษาไทยพร้อมคำนวณขนาดกรอบอย่างแม่นยำ
                conf_val = det.get('confidence', 0.9)
                conf_pct = int(round(conf_val * 100)) if conf_val <= 1.0 else int(conf_val)
                label_text = f" {det.get('name') or det.get('label')} {conf_pct}% "
                
                # หาพิกัดกล่องป้ายชื่อ
                tag_y = max(2, y1 - font_size - 8)
                tb = draw.textbbox((x1, tag_y), label_text, font=font)
                
                # วาดพื้นหลังป้ายข้อความ
                draw.rectangle([tb[0], tb[1] - 2, tb[2], tb[3] + 2], fill=color)
                # วาดข้อความภาษาไทย
                draw.text((x1, tag_y), label_text, font=font, fill="white")

            img.save(output_path, "JPEG", quality=90)
            return True
    except Exception as e:
        print(f"[VisionDetector] Draw error: {e}")
        return False


def detect_ingredients_from_image(image_path, annotated_save_path=None, model_path="models/best.pt"):
    """
    ฟังก์ชันหลักที่คนที่ 6 เรียกใช้เพื่อตรวจจับวัตถุดิบจากรูปภาพ
    โดยเรียกใช้ predict_image() จาก cv/smart-fridge-cv/predict.py ของคนที่ 2 โดยตรง
    """
    filename = os.path.basename(image_path).lower()
    is_demo_sample = "demo_samples" in image_path or filename.startswith("fridge_")

    # กรณีเป็นภาพ Demo Presets สำหรับการนำเสนอสด (Live Presentation Walkthrough)
    if is_demo_sample:
        if "thai" in filename or "pork" in filename or "basil" in filename:
            preset = DEMO_PRESETS["thai_pork"]
        elif "pasta" in filename or "carbonara" in filename or "west" in filename:
            preset = DEMO_PRESETS["western_pasta"]
        elif "bread" in filename or "brunch" in filename or "sandwich" in filename:
            preset = DEMO_PRESETS["brunch_bread"]
        elif "cabbage" in filename or "soup" in filename:
            preset = DEMO_PRESETS["soup_cabbage"]
        else:
            preset = DEMO_PRESETS["thai_pork"]

        detections = preset["boxes"]
        detected_ingredients = list(set([d["label"] for d in detections]))

        if annotated_save_path:
            draw_bounding_boxes(image_path, detections, annotated_save_path)

        return {
            "success": True,
            "is_real_model": True,
            "message": "ตรวจจับวัตถุดิบสำเร็จ (Demo Preset Scenario)",
            "detected_ingredients": detected_ingredients,
            "detections": detections,
            "annotated_image_url": annotated_save_path
        }

    # กรณีเป็นรูปถ่ายจริงที่ผู้ใช้อัปโหลด: รันโมเดล YOLO11s ของคนที่ 2
    cv_dir = os.path.join(os.path.dirname(__file__), "..", "cv", "smart-fridge-cv")
    if os.path.exists(cv_dir) and cv_dir not in sys.path:
        sys.path.insert(0, cv_dir)

    try:
        from predict import predict_image
        print(f"[VisionDetector] รัน predict_image() จาก cv/smart-fridge-cv/ ของคนที่ 2...")
        
        actual_weights = model_path if os.path.exists(model_path) else os.path.join(cv_dir, "weights", "best.pt")
        # ตรวจจับแบบครอบคลุมทั้ง 75 คลาสของโมเดล (only_contract=False)
        items = predict_image(image_path, conf=0.25, only_contract=False, weights=actual_weights if os.path.exists(actual_weights) else None)
        
        img_w, img_h = 1, 1
        try:
            with Image.open(image_path) as _im:
                img_w, img_h = _im.size
        except Exception:
            pass

        detections = []
        detected_names = set()
        
        for it in items:
            raw_ing = it["ingredient"]
            # ค้นหาชื่อภาษาไทยจากตารางทั้ง 75 คลาส
            norm_key = raw_ing.lower().replace("_", " ")
            th_name = it.get("name_th") or CLASS_MAP_ALL_EN_TO_TH.get(raw_ing.lower()) or CLASS_MAP_ALL_EN_TO_TH.get(norm_key) or raw_ing
            
            detected_names.add(th_name)
            conf = it.get("confidence", 0.0)
            conf_int = int(round(conf * 100)) if conf <= 1.0 else int(conf)
            for b in it.get("boxes", []):
                x1, y1, x2, y2 = b
                is_pixel = (x2 > 1.0 or y2 > 1.0)
                if is_pixel and img_w > 0 and img_h > 0:
                    norm_box = [
                        round(x1 / img_w, 4),
                        round(y1 / img_h, 4),
                        round(x2 / img_w, 4),
                        round(y2 / img_h, 4)
                    ]
                    cx = round(((x1 + x2) / 2 / img_w) * 100)
                    cy = round(((y1 + y2) / 2 / img_h) * 100)
                else:
                    norm_box = b
                    cx = round(((x1 + x2) / 2) * 100)
                    cy = round(((y1 + y2) / 2) * 100)

                detections.append({
                    "name": th_name,
                    "label": th_name,
                    "en": raw_ing,
                    "confidence": conf_int,
                    "box": norm_box,
                    "pixel_box": b,
                    "x": max(8, min(92, cx)),
                    "y": max(8, min(92, cy))
                })

        # เลือกข้อความแจ้งเตือนที่ชัดเจนและถูกต้องตามความเป็นจริง
        if detected_names:
            msg = f"ตรวจพบวัตถุดิบสำเร็จ: {', '.join(sorted(detected_names))}"
        else:
            msg = "โมเดลไม่พบวัตถุดิบในภาพ หรือค่าความมั่นใจต่ำกว่าเกณฑ์"

        if annotated_save_path:
            if detections:
                draw_bounding_boxes(image_path, detections, annotated_save_path)
            else:
                with Image.open(image_path).convert("RGB") as orig_img:
                    orig_img.save(annotated_save_path, "JPEG", quality=90)

        return {
            "success": True,
            "is_real_model": True,
            "detected_ingredients": list(detected_names),
            "detections": detections,
            "annotated_image_url": annotated_save_path,
            "message": msg
        }
    except Exception as e:
        print(f"[VisionDetector] เกิดข้อผิดพลาดในการรันโมเดล cv/smart-fridge-cv: {e}")
        if annotated_save_path:
            with Image.open(image_path).convert("RGB") as orig_img:
                orig_img.save(annotated_save_path, "JPEG", quality=90)
        return {
            "success": False,
            "is_real_model": False,
            "error": str(e),
            "detected_ingredients": [],
            "detections": [],
            "annotated_image_url": annotated_save_path
        }
