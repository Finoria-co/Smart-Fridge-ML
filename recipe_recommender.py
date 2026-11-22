import os
import pickle
from functools import lru_cache

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL = os.path.join(_DIR, 'recipe_model.pkl')

@lru_cache(maxsize=1)
def _load():
    with open(_PKL, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['knn_model'], bundle['df'], bundle['ingredient_keywords']


def recommend(user_ingredients: list[str], top_n: int = 5) -> list[dict]:

    knn_model, df, ingredient_keywords = _load()

    user_vector = [
        1 if any(kw in ing for ing in user_ingredients for kw in keywords) else 0
        for keywords in ingredient_keywords.values()
    ]

    distances, indices = knn_model.kneighbors([user_vector], n_neighbors=top_n)

    _col_to_name = {
        'has_pork': 'หมู',         'has_chicken': 'ไก่',       'has_shrimp': 'กุ้ง',
        'has_bacon': 'เบคอน',      'has_egg': 'ไข่',            'has_rice': 'ข้าว',
        'has_pasta': 'พาสต้า',     'has_bread': 'ขนมปัง',      'has_holy_basil': 'กะเพรา',
        'has_tomato': 'มะเขือเทศ', 'has_cheese': 'ชีส',         'has_cabbage': 'ผักกาด',
        'has_fish': 'ปลา',         'has_beef': 'เนื้อ',         'has_shellfish': 'อาหารทะเล',
        'has_noodle': 'เส้น',      'has_coconut_milk': 'กะทิ', 'has_mushroom': 'เห็ด',
    }

    feature_cols = list(ingredient_keywords.keys())
    user_has = {
        col for col, keywords in ingredient_keywords.items()
        if any(kw in ing for ing in user_ingredients for kw in keywords)
    }

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        recipe_needs = [col for col in feature_cols if row[col] == 1]
        have    = [_col_to_name[c] for c in recipe_needs if c in user_has]
        missing = [_col_to_name[c] for c in recipe_needs if c not in user_has]
        match_pct = round(len(have) / len(recipe_needs) * 100, 1) if recipe_needs else 0.0

        results.append({
            'recipe_id': row['recipe_id'],
            'name_th':   row['name_th'],
            'name_en':   row['name_en'],
            'cuisine':   row['cuisine'],
            'match_pct': match_pct,
            'have':      have,
            'missing':   missing,
            'distance':  round(float(dist), 4),
        })

    results.sort(key=lambda r: (-r['match_pct'], r['distance']))
    return results
