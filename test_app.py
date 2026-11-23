import os
from app import app

client = app.test_client()

def test_home():
    res = client.get("/")
    assert res.status_code == 200
    print("✅ GET /: OK (200)")

def test_detect_image():
    import io
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    res = client.post("/api/detect-image", data={"image": (buf, "test.jpg")}, content_type="multipart/form-data")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    print("✅ POST /api/detect-image (Upload): OK")

def test_extract_text():
    res = client.post("/api/extract-text", json={"text": "มีหมูบด 2 ขีด กับไข่ไก่ 3 ฟอง และเส้นสปาเก็ตตี้"})
    assert res.status_code == 200
    data = res.get_json()
    assert "หมูสับ" in data["extracted_ingredients"]
    assert "ไข่ไก่" in data["extracted_ingredients"]
    assert "เส้นพาสต้า" in data["extracted_ingredients"]
    print("✅ POST /api/extract-text: OK - Extracted:", data["extracted_ingredients"])

def test_recommend():
    # 1. ทดสอบ Weighted Cosine Model
    res = client.post("/api/recommend", json={"ingredients": ["หมูสับ", "ไข่ไก่", "ใบกะเพรา"], "cuisine": "ทั้งหมด", "model": "cosine"})
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["recipes"]) > 0
    top_recipe = data["recipes"][0]
    print(f"✅ POST /api/recommend (Cosine): OK - Top recipe: {top_recipe['name']} (Match {top_recipe['match_score']}%)")

    # 2. ทดสอบ Person 5 KNN Model
    res_knn = client.post("/api/recommend", json={"ingredients": ["หมูสับ", "ไข่ไก่"], "cuisine": "ทั้งหมด", "model": "knn"})
    assert res_knn.status_code == 200
    data_knn = res_knn.get_json()
    assert len(data_knn["recipes"]) > 0
    print(f"✅ POST /api/recommend (Person 5 KNN): OK - Top recipe: {data_knn['recipes'][0]['name']} (Match {data_knn['recipes'][0]['match_score']}%)")

    # 3. ทดสอบ Person 5 TF-IDF Model
    res_tfidf = client.post("/api/recommend", json={"ingredients": ["หมูสับ", "ไข่ไก่"], "cuisine": "ทั้งหมด", "model": "tfidf"})
    assert res_tfidf.status_code == 200
    data_tfidf = res_tfidf.get_json()
    assert len(data_tfidf["recipes"]) > 0
    print(f"✅ POST /api/recommend (Person 5 TF-IDF): OK - Top recipe: {data_tfidf['recipes'][0]['name']} (Match {data_tfidf['recipes'][0]['match_score']}%)")

    # 4. ทดสอบ Person 5 Random Forest Model
    res_rf = client.post("/api/recommend", json={"ingredients": ["หมูสับ", "ไข่ไก่"], "cuisine": "ทั้งหมด", "model": "rf"})
    assert res_rf.status_code == 200
    data_rf = res_rf.get_json()
    assert len(data_rf["recipes"]) > 0
    print(f"✅ POST /api/recommend (Person 5 Random Forest): OK - Top recipe: {data_rf['recipes'][0]['name']}")

def test_model_comparison():
    res = client.get("/api/model-comparison")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["models"]) == 3
    print("✅ GET /api/model-comparison: OK - Models count:", len(data["models"]))

if __name__ == "__main__":
    print("--- Running Verification Tests for Member 6 Integration ---")
    test_home()
    test_detect_image()
    test_extract_text()
    test_recommend()
    test_model_comparison()
    print("--- ALL INTEGRATION TESTS PASSED! ---")
