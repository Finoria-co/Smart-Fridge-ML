import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_model(path='models/model.pkl'):
    """โหลด bundle โมเดลของคนที่ 5 (รองรับทั้ง model.pkl และ recipe_model.pkl)"""
    if not os.path.exists(path):
        alt_path = path.replace('model.pkl', 'recipe_model.pkl') if 'model.pkl' in path else path.replace('recipe_model.pkl', 'model.pkl')
        if os.path.exists(alt_path):
            path = alt_path
    with open(path, 'rb') as f:
        return pickle.load(f)

def recommend(user_ingredients, bundle, model='knn', top_n=5):
    """
    ฟังก์ชันแนะนำเมนูอาหารของคนที่ 5 (รองรับ 18 วัตถุดิบเวอร์ชันล่าสุด):
    รองรับโมเดล KNN, TF-IDF และ Random Forest
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

    if model == 'knn' or not (vectorizer or rf_model):
        distances, indices = knn_model.kneighbors([user_vector], n_neighbors=min(top_n, len(df)))
        top_indices = indices[0]
        dists = distances[0]
    elif model == 'tfidf' and vectorizer is not None and tfidf_matrix is not None:
        user_str    = ';'.join(user_ingredients)
        user_vec    = vectorizer.transform([user_str])
        scores      = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        dists = [1.0 - s for s in scores[top_indices]]
    elif model == 'rf' and rf_model is not None:
        proba       = rf_model.predict_proba([user_vector])[0]
        top_indices = proba.argsort()[::-1][:top_n]
        dists = [1.0 - p for p in proba[top_indices]]
    else:
        distances, indices = knn_model.kneighbors([user_vector], n_neighbors=min(top_n, len(df)))
        top_indices = indices[0]
        dists = distances[0]

    feature_cols = list(ingredient_keywords.keys())
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
        row          = df.iloc[idx]
        recipe_needs = [c for c in feature_cols if row.get(c, 0) == 1]
        have         = [_col_to_name.get(c, c) for c in recipe_needs if c in user_has]
        missing      = [_col_to_name.get(c, c) for c in recipe_needs if c not in user_has]
        match_pct    = round(len(have) / len(recipe_needs) * 100, 1) if recipe_needs else 0.0

        dist_val = dists[d_idx] if d_idx < len(dists) else 0.0

        results.append({
            'recipe_id': row.get('recipe_id', ''),
            'name_th':   row.get('name_th', ''),
            'name_en':   row.get('name_en', ''),
            'cuisine':   row.get('cuisine', ''),
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

    results.sort(key=lambda r: (-r['match_pct'], r['distance']))
    return results

