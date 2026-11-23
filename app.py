"""
Smart Fridge Recipe Recommender — Web Backend (สำหรับคนที่ 6: Web UI & Integration)
Framework: Flask
หน้าที่: รวมระบบทั้งหมด (Vision Detector, Text NLP, Recipe ML) เป็น RESTful API และเชื่อมหน้าเว็บ
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory

# นำเข้าโมดูลของเพื่อนร่วมทีมทั้ง 3 คน
from modules.vision_detector import detect_ingredients_from_image, CLASS_NAMES_TH
from modules.text_extractor import extract_ingredients_from_text
from modules.recipe_recommender import recommender

# โหลดโมเดล Machine Learning ของคนที่ 5 (KNN, TF-IDF, Random Forest)
person5_bundle = None
try:
    from modules.recipe_utils import load_model as load_p5_model, recommend as recommend_p5
    p5_model_path = os.path.join("models", "recipe_model.pkl")
    if os.path.exists(p5_model_path):
        person5_bundle = load_p5_model(p5_model_path)
        print("[App] โหลดโมเดลของคนที่ 5 (recipe_model.pkl) สำเร็จ!")
except Exception as e:
    print(f"[App] แจ้งเตือน: ไม่สามารถโหลดโมเดลของคนที่ 5 ได้: {e}")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.json.ensure_ascii = False

try:
    from flask_cors import CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
except ImportError:
    pass

UPLOAD_FOLDER = os.path.join("static", "uploads")
DEMO_FOLDER = os.path.join("static", "demo_samples")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEMO_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """แสดงหน้าเว็บหลัก"""
    return render_template("index.html", standard_classes=CLASS_NAMES_TH)


@app.route("/api/detect-image", methods=["POST"])
def api_detect_image():
    """
    API สำหรับ Step 1 & 2A (คนที่ 2 - Vision ML):
    รับภาพถ่ายตู้เย็นจาก File Upload หรือ Demo Preset
    ส่งตรวจจับ Bounding Box และคืนค่ารายชื่อวัตถุดิบที่ตรวจพบ
    """
    try:
        sample_name = request.form.get("sample_name")
        image_path = None
        unique_id = uuid.uuid4().hex[:8]

        # กรณีเลือกภาพจาก Demo Presets
        if sample_name:
            preset_file = os.path.join(DEMO_FOLDER, sample_name)
            if os.path.exists(preset_file):
                image_path = preset_file
            else:
                return jsonify({"success": False, "error": f"ไม่พบไฟล์เดโม่ {sample_name}"}), 404
        
        # กรณีอัปโหลดไฟล์ใหม่ผ่าน File Upload / Drag & Drop
        elif "image" in request.files or "file" in request.files:
            file = request.files.get("image") or request.files.get("file")
            if file.filename == "":
                return jsonify({"success": False, "error": "ไม่ได้เลือกไฟล์รูปภาพ"}), 400
            
            if file and allowed_file(file.filename):
                ext = file.filename.rsplit(".", 1)[1].lower()
                save_filename = f"upload_{unique_id}.{ext}"
                save_path = os.path.join(UPLOAD_FOLDER, save_filename)
                file.save(save_path)
                image_path = save_path
            else:
                return jsonify({"success": False, "error": "รูปแบบไฟล์ไม่ถูกต้อง (รองรับ .jpg, .png, .webp)"}), 400
        else:
            return jsonify({"success": False, "error": "ไม่พบข้อมูลรูปภาพที่ส่งมา"}), 400

        # วาดกรอบและบันทึกภาพ Annotated
        annotated_filename = f"annotated_{unique_id}.jpg"
        annotated_save_path = os.path.join(UPLOAD_FOLDER, annotated_filename)

        # เรียกใช้โมดูลของคนที่ 2
        result = detect_ingredients_from_image(image_path, annotated_save_path=annotated_save_path)
        if not result.get("success", True):
            return jsonify({
                "success": False,
                "error": result.get("error", "ไม่สามารถตรวจจับวัตถุดิบได้"),
                "detected_ingredients": [],
                "detections": []
            }), 500

        # จัดเตรียม URL สำหรับเรียกแสดงผลบนหน้าเว็บ
        annotated_url = f"/static/uploads/{annotated_filename}" if os.path.exists(annotated_save_path) else None

        return jsonify({
            "success": True,
            "is_real_model": result.get("is_real_model", False),
            "message": result.get("message", "ตรวจจับวัตถุดิบสำเร็จ"),
            "detected_ingredients": result.get("detected_ingredients", []),
            "detections": result.get("detections", []),
            "annotated_image_url": annotated_url
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/extract-text", methods=["POST"])
def api_extract_text():
    """
    API สำหรับ Step 1 & 2B (คนที่ 4 - Text NLP):
    รับข้อความพิมพ์ภาษาไทย สกัดคำวัตถุดิบและแปลงคำพ้อง (Synonyms)
    """
    try:
        data = request.get_json() or {}
        raw_text = data.get("text", "")
        if not raw_text.strip():
            return jsonify({"success": False, "error": "กรุณาพิมพ์ข้อความวัตถุดิบ"}), 400

        # เรียกใช้โมดูลของคนที่ 4
        result = extract_ingredients_from_text(raw_text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    API สำหรับ Step 3 & 4 (คนที่ 5 - Core Recipe ML):
    รับ List วัตถุดิบของผู้ใช้ (ที่ยืนยันแล้ว) -> คำนวณความคล้ายคลึง -> คืนค่าเมนูและสูตร
    รองรับโมเดล: 'cosine' (Weighted Cosine), 'knn', 'tfidf', 'rf' (Random Forest ของคนที่ 5)
    """
    try:
        data = request.get_json() or {}
        ingredients = data.get("ingredients", [])
        cuisine = data.get("cuisine", "ทั้งหมด")
        model_type = data.get("model", "cosine")
        top_k = int(data.get("top_k", 6))

        if not ingredients:
            return jsonify({
                "success": True,
                "recipes": [],
                "count": 0,
                "message": "ไม่มีวัตถุดิบที่เลือก"
            })

        # กรณีเลือกโมเดลของคนที่ 5 (KNN, TF-IDF, Random Forest)
        if model_type in ["knn", "tfidf", "rf"] and person5_bundle:
            raw_p5 = recommend_p5(ingredients, person5_bundle, model=model_type, cuisine_filter=cuisine, top_n=50)
            
            # ทำ Dictionary สำหรับดึงข้อมูลวิธีปรุง (steps)
            recipe_lookup = {r["id"]: r for r in recommender.recipes}
            recipe_lookup_by_name = {r["name"]: r for r in recommender.recipes}

            recipes = []
            for item in raw_p5:
                rec_info = recipe_lookup.get(item.get("recipe_id")) or recipe_lookup_by_name.get(item.get("name_th"))
                steps = rec_info["steps"] if rec_info else [f"เตรียมวัตถุดิบ: {', '.join(item.get('have', []))}", "ปรุงอาหารตามสูตรต้นตำรับ"]
                raw_ings = rec_info.get("raw_ingredients", []) if rec_info else (item.get("have", []) + item.get("missing", []))

                formatted = {
                    "id": item.get("recipe_id", ""),
                    "name": item.get("name_th", ""),
                    "name_en": item.get("name_en", ""),
                    "cuisine": item.get("cuisine", "ไทย"),
                    "origin": rec_info["origin"] if rec_info else "",
                    "course": rec_info["course"] if rec_info else "จานหลัก",
                    "cooking_method": item.get("cooking_method", "ผัด"),
                    "prep_time": rec_info["prep_time"] if rec_info else f"{item.get('total_time_min', 15)} นาที",
                    "calories": f"{item.get('calories_kcal', '-')} kcal",
                    "difficulty": item.get("difficulty", "ง่าย"),
                    "spice_level": rec_info["spice_level"] if rec_info else 0,
                    "flavor_profile": rec_info["flavor_profile"] if rec_info else "",
                    "description": item.get("description", ""),
                    "match_score": int(round(item.get("match_pct", 0))),
                    "cosine_similarity": round(max(0.0, 1.0 - float(item.get("distance", 0.0))), 4),
                    "matched_ingredients": item.get("have", []),
                    "missing_ingredients": item.get("missing", []),
                    "is_ready_to_cook": len(item.get("missing", [])) == 0,
                    "seasonings": raw_ings,
                    "steps": steps
                }
                recipes.append(formatted)

            # เรียงลำดับจาก match_score มากไปน้อยอย่างเคร่งครัดเหมือน Weighted Cosine
            recipes.sort(key=lambda x: (
                x["is_ready_to_cook"],
                x["match_score"],
                -len(x["missing_ingredients"]),
                x["cosine_similarity"]
            ), reverse=True)
            recipes = recipes[:top_k]
        else:
            # กรณีโมเดลหลัก Weighted Cosine Similarity
            recipes = recommender.recommend(ingredients, cuisine_filter=cuisine, top_k=top_k)

        return jsonify({
            "success": True,
            "recipes": recipes,
            "count": len(recipes),
            "cuisine_filter": cuisine,
            "model_type": model_type,
            "input_ingredients": ingredients
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model-comparison", methods=["GET"])
def api_model_comparison():
    """API ดึงข้อมูลเปรียบเทียบโมเดล 3 ตัวสำหรับใช้โชว์อาจารย์ตอนพรีเซนต์"""
    comparison_data = recommender.get_model_comparison_metrics()
    return jsonify({
        "success": True,
        "models": comparison_data
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Smart Fridge Recipe Recommender Backend (Flask)")
    print("📌 พัฒนาโดย: คนที่ 6 (Web UI & Integration)")
    print("🔗 เข้าใช้งานเว็บได้ที่: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
