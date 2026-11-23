"""ส่งวัตถุดิบที่ YOLO ทายได้ (results/demo_pick/candidates.csv) เข้าตัวแนะนำเมนูของคนที่ 5
จำลอง recipe_recommender.py ของ fridge-chef-ai ด้วย numpy (KNN cosine บน feature คีย์เวิร์ด 18 ตัว
เหมือน model.ipynb) เพราะ venv นี้ไม่มี sklearn · NearestNeighbors(metric='cosine') = brute force cosine distance

ชื่อไทยที่ส่งเข้าใช้ตารางเดียวกับเว็บรวม (Smart-Fridge-ML/modules/vision_detector.py)
แสดง 2 แบบ: buggy = เหมือน recommend() ตอนนี้ ("ไข่ไก่" ติด has_chicken ด้วย) · fixed = กันกรณีไข่แบบใน model.ipynb

    python scripts/demo_recommend.py <recipes_dataset.csv> [--top 30]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STD_TO_SYSTEM_TH = {  # Smart-Fridge-ML/modules/vision_detector.py CLASS_MAP_EN_TO_TH
    "pork": "หมูสับ", "chicken": "อกไก่", "shrimp": "กุ้งสด", "bacon": "เบคอน", "egg": "ไข่ไก่",
    "bread": "ขนมปัง", "holy_basil": "ใบกะเพรา", "tomato": "มะเขือเทศ", "cheese": "ชีส", "cabbage": "ผักกาดขาว",
}
KW = {  # fridge-chef-ai/model.ipynb
    'has_pork': ['หมูสับ', 'หมู'], 'has_chicken': ['อกไก่', 'ไก่'], 'has_shrimp': ['กุ้ง'],
    'has_bacon': ['เบคอน', 'แฮม'], 'has_egg': ['ไข่'], 'has_rice': ['ข้าว'],
    'has_pasta': ['พาสต้า', 'สปาเก็ต', 'เพนเน่', 'ลาซานญ่า'], 'has_bread': ['ขนมปัง'],
    'has_holy_basil': ['กะเพรา'], 'has_tomato': ['มะเขือเทศ'], 'has_cheese': ['ชีส'],
    'has_cabbage': ['ผักกาด'], 'has_fish': ['ปลากะพง', 'ปลาค็อด', 'ปลาแซลมอน', 'ปลาหมึก', 'ปลา'],
    'has_beef': ['เนื้อวัว', 'เนื้อริบอาย', 'เนื้อสันคอ', 'เนื้อแกะ'],
    'has_shellfish': ['ปูม้า', 'เนื้อปู', 'หอยลาย', 'หอยแมลงภู่'],
    'has_noodle': ['เส้นจันท์', 'เส้นใหญ่', 'วุ้นเส้น', 'บะหมี่'],
    'has_coconut_milk': ['กะทิ', 'น้ำกะทิ'], 'has_mushroom': ['เห็ด'],
}
COLS = list(KW)


def has(items, kws, egg_guard):
    return any(kw in it for it in items for kw in kws
               if not (egg_guard and it.strip().startswith('ไข่') and kw == 'ไก่'))


def load_recipes(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    X = np.array([[int(has(r['main_ingredients'].split(';'), KW[c], True)) for c in COLS] for r in rows], float)
    return rows, X


def recommend(user, rows, X, egg_guard, top_n=5):
    u = np.array([int(has(user, KW[c], egg_guard)) for c in COLS], float)
    sim = X @ u / (np.linalg.norm(X, axis=1) * np.linalg.norm(u) + 1e-12)
    idx = np.argsort(1 - sim, kind='stable')[:top_n]
    out = []
    for i in idx:
        need = [c for c in COLS if X[i, list(COLS).index(c)] == 1]
        hv = [c for c in need if u[COLS.index(c)] == 1]
        out.append((rows[i]['recipe_id'], rows[i]['name_th'], round(len(hv) / len(need) * 100, 1) if need else 0.0,
                    round(float(1 - sim[i]), 3)))
    out.sort(key=lambda r: (-r[2], r[3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recipes')
    ap.add_argument('--top', type=int, default=30)
    a = ap.parse_args()
    rows, X = load_recipes(a.recipes)
    cands = [r for r in csv.DictReader(open(ROOT / 'results/demo_pick/candidates.csv', encoding='utf8'))
             if r['exact'] == '1' and int(r['n_gt']) >= 2 and r['source'] != 'foodseg103']
    cands.sort(key=lambda r: -float(r['min_conf']))
    res = []
    for c in cands[:a.top]:
        th = [STD_TO_SYSTEM_TH[s] for s in c['pred'].split(';')]
        b = recommend(th, rows, X, egg_guard=False)[0]
        f = recommend(th, rows, X, egg_guard=True)[0]
        res.append({**c, 'input_th': th, 'top1_buggy': b, 'top1_fixed': f})
        print(f"{c['file'][:60]:60} {c['source']:14} minconf {c['min_conf']} | {','.join(th)}\n"
              f"    buggy: {b[1]} ({b[0]}) {b[2]}% · fixed: {f[1]} ({f[0]}) {f[2]}%")
    (ROOT / 'results/demo_pick/recommend_top.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
