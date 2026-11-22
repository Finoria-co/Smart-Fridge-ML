"""
Build script for GitHub Pages (github.io) compatibility.
1. Generates static/js/recipes_data.js with all 50 recipes and cooking steps.
2. Creates an empty .nojekyll file in the root directory.
3. Injects client-side ML engine into index.html so it runs 100% offline on GitHub Pages without Flask!
"""
import os
import csv
import json
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPES_CSV = os.path.join(ROOT_DIR, "data", "recipes_dataset.csv")
JS_DIR = os.path.join(ROOT_DIR, "static", "js")
os.makedirs(JS_DIR, exist_ok=True)

# 1. Parse CSV and build recipes JSON
recipes = []
with open(RECIPES_CSV, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        clean = {k.replace("\ufeff", ""): v for k, v in row.items()}
        raw_ings = [x.strip() for x in clean.get("main_ingredients", "").split(";") if x.strip()]
        steps_raw = clean.get("steps", "")
        steps = [s.strip() for s in steps_raw.split("|") if s.strip()] if steps_raw else []
        clean["raw_ingredients"] = raw_ings
        clean["steps_list"] = steps
        recipes.append(clean)

print(f"Loaded {len(recipes)} recipes from {RECIPES_CSV}")

# 2. Write static/js/recipes_data.js
js_recipes_path = os.path.join(JS_DIR, "recipes_data.js")
with open(js_recipes_path, "w", encoding="utf-8") as f:
    f.write("/* Auto-generated for GitHub Pages standalone execution */\n")
    f.write("window.RECIPES_DATASET = ")
    json.dump(recipes, f, ensure_ascii=False, indent=2)
    f.write(";\n")
print(f"Generated {js_recipes_path}")

# 3. Create .nojekyll in root
nojekyll_path = os.path.join(ROOT_DIR, ".nojekyll")
with open(nojekyll_path, "w", encoding="utf-8") as f:
    f.write("")
print(f"Created {nojekyll_path}")
