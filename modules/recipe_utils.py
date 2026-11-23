import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def load_model(path='models/model.pkl'):
    """โหลด bundle โมเดลของคนที่ 5 (รองรับทั้ง model.pkl และ recipe_model.pkl)"""
    if not os.path.exists(path):
        alt_path = path.replace('model.pkl', 'recipe_model.pkl') if 'model.pkl' in path else path.replace('recipe_model.pkl', 'model.pkl')
        if os.path.exists(alt_path):
            path = alt_path
    with open(path, 'rb') as f:
        bundle = pickle.load(f)

    df = bundle.get('df')

    # ซิงค์ข้อมูลเมนูอาหารกับ data/recipes_dataset.csv (รองรับการอัปเดตเมนู 80 สูตรอาหาร)
    recipes_csv_path = 'data/recipes_dataset.csv'
    if os.path.exists(recipes_csv_path):
        try:
            import pandas as pd
            csv_df = pd.read_csv(recipes_csv_path)
            if df is None or len(csv_df) > len(df):
                ingredient_keywords = bundle.get('ingredient_keywords', {})
                for col, kws in ingredient_keywords.items():
                    if col not in csv_df.columns:
                        csv_df[col] = csv_df['main_ingredients'].fillna('').apply(
                            lambda ing_text: 1 if any(kw in str(ing_text) for kw in kws) else 0
                        )
                bundle['df'] = csv_df
                df = csv_df
                from sklearn.neighbors import NearestNeighbors
                feature_cols = [c for c in ingredient_keywords.keys() if c in df.columns]
                if feature_cols:
                    knn = NearestNeighbors(metric='cosine', n_neighbors=min(5, len(df)))
                    knn.fit(df[feature_cols].values)
                    bundle['knn_model'] = knn
                bundle['vectorizer'] = None
                bundle['tfidf_matrix'] = None
                bundle['rf_model'] = None
        except Exception as e:
            print(f"[recipe_utils] Warning during recipe sync: {e}")

    # หาก bundle เดิมของคนที่ 5 ยังไม่มี vectorizer หรือ rf_model ให้สร้างเสริมเข้าไปอัตโนมัติ เพื่อให้เลือกใช้ได้จริงครบทุกโมเดล
    if df is not None:
        if bundle.get('vectorizer') is None or bundle.get('tfidf_matrix') is None:
            corpus = (df['main_ingredients'].fillna('') + ' ' + df['name_th'].fillna('') + ' ' + df.get('description', '').fillna(''))
            vec = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
            bundle['tfidf_matrix'] = vec.fit_transform(corpus)
            bundle['vectorizer'] = vec

        if bundle.get('rf_model') is None:
            ingredient_keywords = bundle.get('ingredient_keywords', {})
            feature_cols = [c for c in ingredient_keywords.keys() if c in df.columns]
            if feature_cols:
                X = df[feature_cols].values
                y = df['cuisine'].fillna('Thai').values
                rf = RandomForestClassifier(n_estimators=30, random_state=42)
                rf.fit(X, y)
                bundle['rf_model'] = rf

    return bundle

def recommend(user_ingredients, bundle, model='knn', cuisine_filter='ทั้งหมด', top_n=6):
    """
    ฟังก์ชันแนะนำเมนูอาหารของคนที่ 5 (รองรับ 18 วัตถุดิบเวอร์ชันล่าสุด):
    รองรับโมเดล KNN, TF-IDF และ Random Forest พร้อมการกรองสัญชาติอาหาร (cuisine_filter)
    """
    df                  = bundle['df']
    ingredient_keywords = bundle['ingredient_keywords']
    knn_model           = bundle.get('knn_model')
    rf_model            = bundle.get('rf_model')
    vectorizer          = bundle.get('vectorizer')
    tfidf_matrix        = bundle.get('tfidf_matrix')

    # สร้าง user vector (binary 18 มิติ)
    user_vector = [
        1 if any(
            kw in ing
            for ing in user_ingredients
            for kw in kws
            if not (ing.strip().startswith('ไข่') and kw == 'ไก่')
        )
        else 0
        for kws in ingredient_keywords.values()
    ]

    _col_to_name = {
        'has_pork': 'หมู',         'has_chicken': 'ไก่',       'has_shrimp': 'กุ้ง',
        'has_bacon': 'เบคอน',      'has_egg': 'ไข่',            'has_rice': 'ข้าว',
        'has_pasta': 'พาสต้า',     'has_bread': 'ขนมปัง',      'has_holy_basil': 'กะเพรา',
        'has_tomato': 'มะเขือเทศ', 'has_cheese': 'ชีส',         'has_cabbage': 'ผักกาด',
        'has_fish': 'ปลา',         'has_beef': 'เนื้อ',         'has_shellfish': 'อาหารทะเล',
        'has_noodle': 'เส้น',      'has_coconut_milk': 'กะทิ', 'has_mushroom': 'เห็ด',
    }

    feature_cols = list(ingredient_keywords.keys())

    # ประเมินคะแนนทั้ง 50 เมนู เพื่อให้กรองสัญชาติอาหารได้อย่างสมบูรณ์แบบ
    total_recipes = len(df)
    if model == 'knn' or (not vectorizer and not rf_model):
        distances, indices = knn_model.kneighbors([user_vector], n_neighbors=total_recipes)
        top_indices = indices[0]
        dists = distances[0]
    elif model == 'tfidf' and vectorizer is not None and tfidf_matrix is not None:
        user_str    = ' '.join(user_ingredients)
        user_vec    = vectorizer.transform([user_str])
        scores      = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1]
        dists = [1.0 - s for s in scores[top_indices]]
    elif model == 'rf' and rf_model is not None:
        importances = rf_model.feature_importances_
        X = df[feature_cols].values
        # คำนวณความคล้ายคลึงตามน้ำหนักความสำคัญของต้นไม้ตัดสินใจ (Tree Importance Scoring)
        scores = (np.array(user_vector) * importances) @ X.T
        top_indices = scores.argsort()[::-1]
        dists = [1.0 - (scores[i] / (sum(importances) + 1e-5)) for i in top_indices]
    else:
        distances, indices = knn_model.kneighbors([user_vector], n_neighbors=total_recipes)
        top_indices = indices[0]
        dists = distances[0]

    user_has = {
        col for col, keywords in ingredient_keywords.items()
        if any(
            kw in ing 
            for ing in user_ingredients 
            for kw in keywords
            if not (ing.strip().startswith('ไข่') and kw == 'ไก่')
        )
    }

    results = []
    for d_idx, idx in enumerate(top_indices):
        row = df.iloc[idx]
        
        # กรองตามสัญชาติอาหาร (ถ้าเลือก)
        row_c = str(row.get('cuisine', '')).strip().lower()
        if cuisine_filter == 'ไทย' and 'thai' not in row_c and 'ไทย' not in row_c:
            continue
        if cuisine_filter == 'ฝรั่ง' and 'western' not in row_c and 'ฝรั่ง' not in row_c:
            continue

        recipe_needs = [c for c in feature_cols if row.get(c, 0) == 1]
        have         = [_col_to_name.get(c, c) for c in recipe_needs if c in user_has]
        missing      = [_col_to_name.get(c, c) for c in recipe_needs if c not in user_has]
        match_pct    = round(len(have) / len(recipe_needs) * 100, 1) if recipe_needs else 0.0

        dist_val = dists[d_idx] if d_idx < len(dists) else 0.0

        results.append({
            'recipe_id': row.get('recipe_id', ''),
            'name_th':   row.get('name_th', ''),
            'name_en':   row.get('name_en', ''),
            'cuisine':   'ไทย' if ('thai' in row_c or 'ไทย' in row_c) else 'ฝรั่ง',
            'match_pct': match_pct,
            'have':      have,
            'missing':   missing,
            'distance':  round(float(dist_val), 4),
            'description':    row.get('description', ''),
            'cooking_method': row.get('cooking_method', 'ผัด'),
            'prep_time_min':  int(row.get('prep_time_min', 10)),
            'cook_time_min':  int(row.get('cook_time_min', 5)),
            'difficulty':     row.get('difficulty', 'ง่าย'),
            'calories_kcal':  int(row.get('calories_kcal', 400)),
            'allergens':      row.get('allergens', 'ไม่มี'),
        })

    # เรียงลำดับจากความพร้อมปรุง (100% ก่อน), match_pct มากไปน้อย, และขาดวัตถุดิบน้อยที่สุด
    results.sort(key=lambda r: (
        len(r['missing']) == 0,
        r['match_pct'],
        -len(r['missing']),
        -r['distance']
    ), reverse=True)
    return results[:top_n]

