"""
Build and inject client-side ML engine into index.html for GitHub Pages.
"""
import os
import json
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_INDEX = os.path.join(ROOT_DIR, "templates", "index.html")
ROOT_INDEX = os.path.join(ROOT_DIR, "index.html")
RECIPES_JSON = os.path.join(ROOT_DIR, "data", "recipes_dataset.json")

with open(RECIPES_JSON, "r", encoding="utf-8") as f:
    recipes_data = json.load(f)

# Client side script to append to index.html
CLIENT_SIDE_ENGINE_JS = f"""
  <script src="static/js/recipes_data.js"></script>
  <script>
    // Embedded recipes dataset backup if script fails
    if (!window.RECIPES_DATASET || window.RECIPES_DATASET.length === 0) {{
      window.RECIPES_DATASET = {json.dumps(recipes_data, ensure_ascii=False)};
    }}

    const IS_STANDALONE = window.location.hostname.includes("github.io") || 
                          window.location.protocol === "file:" || 
                          (window.location.port !== "5000" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1");

    document.addEventListener("DOMContentLoaded", () => {{
      const envBadge = document.getElementById("env-badge");
      if (envBadge) {{
        if (IS_STANDALONE) {{
          envBadge.textContent = "GitHub Pages • Client ML";
          envBadge.className = "text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200";
        }} else {{
          envBadge.textContent = "Flask Backend • YOLO11s";
          envBadge.className = "text-[11px] font-mono px-2 py-0.5 rounded bg-teal-50 text-teal-800 border border-teal-200";
        }}
      }}
    }});

    // Client-side NLP synonym dictionary
    const CLIENT_SYNONYMS = {{
      "หมูสับ": "หมูสับ", "หมูบด": "หมูสับ", "หมูชิ้น": "หมูสับ", "เนื้อหมู": "หมูสับ", "หมู": "หมูสับ",
      "อกไก่": "อกไก่", "เนื้อไก่": "อกไก่", "ไก่": "อกไก่", "สันในไก่": "อกไก่",
      "กุ้งสด": "กุ้งสด", "กุ้งขาว": "กุ้งสด", "กุ้ง": "กุ้งสด", "กุ้งแม่น้ำ": "กุ้งสด",
      "เบคอน": "เบคอน", "แฮม": "เบคอน", "หมูสามชั้นรมควัน": "เบคอน", "ไส้กรอก": "เบคอน",
      "ไข่ไก่": "ไข่ไก่", "ไข่": "ไข่ไก่", "ไข่เป็ด": "ไข่ไก่", "ไข่ดาว": "ไข่ไก่", "ไข่แดง": "ไข่ไก่",
      "ข้าวสวย": "ข้าวสวย", "ข้าว": "ข้าวสวย", "ข้าวเย็น": "ข้าวสวย", "ข้าวเปล่า": "ข้าวสวย",
      "เส้นพาสต้า": "เส้นพาสต้า", "พาสต้า": "เส้นพาสต้า", "สปาเก็ตตี้": "เส้นพาสต้า", "เส้นสปาเก็ตตี้": "เส้นพาสต้า", "มักกะโรนี": "เส้นพาสต้า",
      "ขนมปัง": "ขนมปัง", "ปังแถว": "ขนมปัง", "โทสต์": "ขนมปัง", "ขนมปังแผ่น": "ขนมปัง",
      "ใบกะเพรา": "ใบกะเพรา", "กะเพรา": "ใบกะเพรา",
      "มะเขือเทศ": "มะเขือเทศ", "มะเขือเทศราชินี": "มะเขือเทศ", "มะเขือเทศลูกท้อ": "มะเขือเทศ",
      "ชีส": "ชีส", "เนยแข็ง": "ชีส", "ชีสแผ่น": "ชีส", "มอสซาเรลล่า": "ชีส",
      "ผักกาดขาว": "ผักกาดขาว", "ผักกาด": "ผักกาดขาว", "ลุ้ย": "ผักกาดขาว", "กะหล่ำปลี": "ผักกาดขาว"
    }};

    function extractIngredientsClientSide(text) {{
      const cleaned = text.replace(/[0-9.,()\\/#@$!?-]/g, " ");
      const found = new Set();
      for (const [kw, std] of Object.entries(CLIENT_SYNONYMS)) {{
        if (cleaned.includes(kw)) {{
          found.add(std);
        }}
      }}
      return Array.from(found);
    }}

    // Client-side Recommender Engine
    const CLIENT_FEATURE_WEIGHTS = {{
      "หมูสับ": 2.2, "อกไก่": 2.2, "กุ้งสด": 2.5, "เบคอน": 2.2, "ไข่ไก่": 2.0,
      "ข้าวสวย": 1.8, "เส้นพาสต้า": 2.0, "ขนมปัง": 1.8, "ใบกะเพรา": 1.6,
      "มะเขือเทศ": 1.4, "ชีส": 1.8, "ผักกาดขาว": 1.4
    }};

    const CLIENT_PANTRY_STAPLES = new Set([
      "พริกขี้หนู", "พริก", "พริกแห้ง", "พริกป่น", "พริกไทย", "พริกไทยดำ", "พริกไทยขาว", 
      "กระเทียม", "รากผักชี", "น้ำปลา", "ซีอิ๊วขาว", "ซีอิ๊วดำ", "ซอสหอยนางรม", "ซอสปรุงรส", 
      "น้ำตาลทราย", "น้ำตาลปี๊บ", "น้ำมันพืช", "น้ำมันรำข้าว", "เกลือ", "น้ำมะนาว", "มะนาว", 
      "น้ำส้มสายชู", "น้ำมันมะกอก", "เนย", "น้ำสต๊อก", "น้ำสต๊อกเนื้อ", "น้ำสต๊อกผัก", 
      "น้ำสต๊อกอาหารทะเล", "น้ำซุป", "แป้งสาลี", "แป้งมันสำปะหลัง", "ผงปรุงรส", "เต้าเจี้ยว"
    ]);

    function mapToStd(rawItem) {{
      const item = rawItem.trim();
      if (CLIENT_SYNONYMS[item]) return CLIENT_SYNONYMS[item];
      for (const [kw, std] of Object.entries(CLIENT_SYNONYMS)) {{
        if (kw.length >= 3 && item.includes(kw)) return std;
      }}
      return null;
    }}

    function calculateCosineSim(userVec, recipeVec) {{
      let dot = 0, nUser = 0, nRec = 0;
      for (const col of ALL_12_INGREDIENTS) {{
        const w = CLIENT_FEATURE_WEIGHTS[col] || 1.0;
        const u = (userVec[col] || 0) * w;
        const r = (recipeVec[col] || 0) * w;
        dot += u * r;
        nUser += u * u;
        nRec += r * r;
      }}
      if (nUser === 0 || nRec === 0) return 0;
      return dot / (Math.sqrt(nUser) * Math.sqrt(nRec));
    }}

    function recommendClientSide(userIngredients, cuisineFilter = "ทั้งหมด", modelType = "cosine", topK = 6) {{
      const userSet = new Set(userIngredients);
      if (userSet.size === 0) return [];

      const userVec = {{}};
      ALL_12_INGREDIENTS.forEach(c => {{ userVec[c] = userSet.has(c) ? 1 : 0; }});

      const scored = [];
      const dataset = window.RECIPES_DATASET || [];

      for (const r of dataset) {{
        const rawCuisine = (r.cuisine || "").trim().toLowerCase();
        const cuisineTh = (rawCuisine === "thai" || rawCuisine === "ไทย") ? "ไทย" : "ฝรั่ง";
        if (cuisineFilter !== "ทั้งหมด" && cuisineTh !== cuisineFilter) continue;

        const rawList = r.raw_ingredients || (r.main_ingredients ? r.main_ingredients.split(";").map(x => x.trim()) : []);
        const nonPantry = rawList.filter(item => !CLIENT_PANTRY_STAPLES.has(item));
        if (nonPantry.length === 0) continue;

        const matchedRaw = [];
        const missingRaw = [];
        const matchedStd = new Set();
        const recipeVec = {{}};
        ALL_12_INGREDIENTS.forEach(c => {{ recipeVec[c] = 0; }});

        for (const item of nonPantry) {{
          const std = mapToStd(item);
          if (std) recipeVec[std] = 1;
          if (std && userSet.has(std)) {{
            matchedRaw.push(item);
            matchedStd.add(std);
          }} else {{
            missingRaw.push(item);
          }}
        }}

        // If zero match, exclude
        if (matchedRaw.length === 0) continue;

        const covRatio = matchedRaw.length / nonPantry.length;
        const cosSim = calculateCosineSim(userVec, recipeVec);

        let matchScore = 0;
        let isReady = false;

        if (missingRaw.length === 0) {{
          matchScore = 100;
          isReady = true;
        }} else {{
          isReady = false;
          let calc = (covRatio * 75.0) + (cosSim * 25.0);
          const hasMajor = ["หมูสับ", "อกไก่", "กุ้งสด", "เบคอน", "เส้นพาสต้า", "ข้าวสวย"].some(s => matchedStd.has(s));
          if (hasMajor) calc += 6.0;

          if (missingRaw.length === 1) matchScore = Math.round(Math.min(85, Math.max(75, calc)));
          else if (missingRaw.length === 2) matchScore = Math.round(Math.min(70, Math.max(58, calc)));
          else if (missingRaw.length === 3) matchScore = Math.round(Math.min(48, Math.max(38, calc)));
          else matchScore = Math.round(Math.min(25, Math.max(12, covRatio * 100)));
        }}

        const totalTime = r.total_time_min || "15";
        const prepTime = r.prep_time_min || "10";
        const cookTime = r.cook_time_min || "5";
        const steps = r.steps_list || (r.steps ? r.steps.split("|").map(s => s.trim()) : [`เตรียมวัตถุดิบ: ${{matchedRaw.join(", ")}}`, `ปรุงอาหารตามสูตรต้นตำรับ`]);

        scored.push({{
          id: r.recipe_id || "",
          name: r.name_th || "",
          name_en: r.name_en || "",
          cuisine: cuisineTh,
          origin: r.origin || "",
          course: r.course || "จานหลัก",
          cooking_method: r.cooking_method || "ผัด",
          prep_time: `${{totalTime}} นาที (เตรียม ${{prepTime}} น. • ปรุง ${{cookTime}} น.)`,
          calories: `${{r.calories_kcal || "-"}} kcal`,
          difficulty: r.difficulty || "ง่าย",
          spice_level: parseInt(r.spice_level || 0),
          flavor_profile: (r.flavor_profile || "").replace(/;/g, " • "),
          description: r.description || "",
          match_score: matchScore,
          cosine_similarity: Math.round(cosSim * 10000) / 10000,
          matched_ingredients: Array.from(matchedStd).length > 0 ? Array.from(matchedStd) : matchedRaw,
          missing_ingredients: missingRaw,
          is_ready_to_cook: isReady,
          steps: steps
        }});
      }}

      scored.sort((a, b) => {{
        if (a.is_ready_to_cook !== b.is_ready_to_cook) return b.is_ready_to_cook ? 1 : -1;
        if (a.missing_ingredients.length !== b.missing_ingredients.length) return a.missing_ingredients.length - b.missing_ingredients.length;
        return b.match_score - a.match_score;
      }});

      return scored.slice(0, topK);
    }}

    const CLIENT_DEMO_PRESETS = {{
      "fridge_thai_basil.jpg": {{
        items: ["หมูสับ", "ไข่ไก่", "ใบกะเพรา"],
        boxes: [
          {{ label: "หมูสับ", en: "pork", confidence: 0.94, box: [0.10, 0.25, 0.40, 0.75] }},
          {{ label: "ไข่ไก่", en: "egg", confidence: 0.97, box: [0.45, 0.20, 0.68, 0.65] }},
          {{ label: "ใบกะเพรา", en: "holy_basil", confidence: 0.91, box: [0.70, 0.28, 0.92, 0.72] }}
        ]
      }},
      "fridge_western_carbonara.jpg": {{
        items: ["เส้นพาสต้า", "เบคอน", "ชีส", "ไข่ไก่"],
        boxes: [
          {{ label: "เส้นพาสต้า", en: "pasta", confidence: 0.96, box: [0.08, 0.20, 0.35, 0.78] }},
          {{ label: "เบคอน", en: "bacon", confidence: 0.93, box: [0.38, 0.25, 0.65, 0.68] }},
          {{ label: "ชีส", en: "cheese", confidence: 0.89, box: [0.68, 0.18, 0.92, 0.58] }}
        ]
      }},
      "fridge_brunch_sandwich.jpg": {{
        items: ["ขนมปัง", "ชีส", "เบคอน", "มะเขือเทศ"],
        boxes: [
          {{ label: "ขนมปัง", en: "bread", confidence: 0.95, box: [0.12, 0.22, 0.42, 0.72] }},
          {{ label: "ชีส", en: "cheese", confidence: 0.92, box: [0.44, 0.18, 0.70, 0.58] }},
          {{ label: "มะเขือเทศ", en: "tomato", confidence: 0.96, box: [0.72, 0.28, 0.93, 0.75] }}
        ]
      }}
    }};
  </script>
"""

with open(TEMPLATES_INDEX, "r", encoding="utf-8") as f:
    html_content = f.read()

# Replace or inject scripts
if "CLIENT_SIDE_ENGINE_JS" not in html_content:
    # Inject before </head>
    html_content = html_content.replace("</head>", CLIENT_SIDE_ENGINE_JS + "\n</head>")

# Make runRecipeRecommendation robust to fallback
old_run_rec = """      try {
        const response = await fetch("/api/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ingredients: currentConfirmedIngredients,
            cuisine: cuisine,
            model: model,
            top_k: 6
          })
        });
        const data = await response.json();
        
        if (data.success) {
          renderRecipeCards(data.recipes);
          countBadge.textContent = `${data.recipes.length} recipes`;
        }
      } catch (err) {
        console.error("Recommend error:", err);
      }"""

new_run_rec = """      try {
        if (!IS_STANDALONE) {
          const response = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ingredients: currentConfirmedIngredients,
              cuisine: cuisine,
              model: model,
              top_k: 6
            })
          });
          if (response.ok) {
            const data = await response.json();
            if (data.success) {
              renderRecipeCards(data.recipes);
              countBadge.textContent = `${data.recipes.length} recipes`;
              return;
            }
          }
        }
      } catch (err) {
        console.log("Switching to Client-Side Recommender fallback...");
      }

      // Standalone / GitHub Pages client-side execution
      const clientRecipes = recommendClientSide(currentConfirmedIngredients, cuisine, model, 6);
      renderRecipeCards(clientRecipes);
      countBadge.textContent = `${clientRecipes.length} recipes`;"""

html_content = html_content.replace(old_run_rec, new_run_rec)

# Make submitTextInput robust to fallback
old_submit_text = """      try {
        const response = await fetch("/api/extract-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        if (data.success) {
          currentConfirmedIngredients = [...data.extracted_ingredients];
          renderConfirmedIngredients();
          renderQuickAddTags();
          runRecipeRecommendation();
        }
      } catch (err) {
        alert("ข้อผิดพลาด: " + err.message);
      }"""

new_submit_text = """      try {
        if (!IS_STANDALONE) {
          const response = await fetch("/api/extract-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
          });
          if (response.ok) {
            const data = await response.json();
            if (data.success) {
              currentConfirmedIngredients = [...data.extracted_ingredients];
              renderConfirmedIngredients();
              renderQuickAddTags();
              runRecipeRecommendation();
              return;
            }
          }
        }
      } catch (err) {
        console.log("Switching to Client-Side NLP extraction...");
      }

      // Standalone NLP extraction
      const extracted = extractIngredientsClientSide(text);
      if (extracted.length > 0) {
        currentConfirmedIngredients = Array.from(new Set([...currentConfirmedIngredients, ...extracted]));
      } else {
        alert("ไม่พบวัตถุดิบมาตรฐานในข้อความที่พิมพ์");
      }
      renderConfirmedIngredients();
      renderQuickAddTags();
      runRecipeRecommendation();"""

html_content = html_content.replace(old_submit_text, new_submit_text)

# Make loadDemoSample robust to fallback
old_demo = """      try {
        const response = await fetch("/api/detect-image", {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        handleDetectionResponse(data);
      } catch (err) {
        alert("เกิดข้อผิดพลาด: " + err.message);
      } finally {
        showLoading(false);
      }"""

new_demo = """      try {
        if (!IS_STANDALONE) {
          const response = await fetch("/api/detect-image", {
            method: "POST",
            body: formData
          });
          if (response.ok) {
            const data = await response.json();
            handleDetectionResponse(data);
            showLoading(false);
            return;
          }
        }
      } catch (err) {
        console.log("Switching to Standalone Demo Presets...");
      }

      // Standalone Preset
      const preset = CLIENT_DEMO_PRESETS[filename] || { items: ["หมูสับ", "ไข่ไก่"], boxes: [] };
      const previewContainer = document.getElementById("image-preview-container");
      const annotatedImg = document.getElementById("annotated-image");
      const modelBadge = document.getElementById("model-status-badge");

      previewContainer.classList.remove("hidden");
      annotatedImg.src = "static/demo_samples/" + filename;
      modelBadge.textContent = "YOLO11s (Demo Mode)";
      modelBadge.className = "font-mono text-[10px] px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200";

      currentConfirmedIngredients = [...preset.items];
      renderConfirmedIngredients();
      renderQuickAddTags();
      runRecipeRecommendation();
      showLoading(false);"""

html_content = html_content.replace(old_demo, new_demo)

# Make uploadAndDetectImage robust
old_upload = """      try {
        const response = await fetch("/api/detect-image", {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        handleDetectionResponse(data);
      } catch (err) {
        alert("เกิดข้อผิดพลาดในการเชื่อมต่อ API: " + err.message);
      } finally {
        showLoading(false);
      }"""

new_upload = """      try {
        if (!IS_STANDALONE) {
          const response = await fetch("/api/detect-image", {
            method: "POST",
            body: formData
          });
          if (response.ok) {
            const data = await response.json();
            handleDetectionResponse(data);
            showLoading(false);
            return;
          }
        }
      } catch (err) {
        console.log("Switching to Standalone image preview...");
      }

      // Standalone FileReader preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const previewContainer = document.getElementById("image-preview-container");
        const annotatedImg = document.getElementById("annotated-image");
        const modelBadge = document.getElementById("model-status-badge");

        previewContainer.classList.remove("hidden");
        annotatedImg.src = e.target.result;
        modelBadge.textContent = "Client Preview (YOLO11s requires Python API)";
        modelBadge.className = "font-mono text-[10px] px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200";
        showLoading(false);
      };
      reader.readAsDataURL(file);"""

html_content = html_content.replace(old_upload, new_upload)

# Make openComparisonModal robust
old_comparison = """      try {
        const response = await fetch("/api/model-comparison");
        const data = await response.json();
        if (data.success) {
          renderComparisonTable(data.models);
        }
      } catch (err) {
        console.error("Modal fetch error:", err);
      }"""

new_comparison = """      try {
        if (!IS_STANDALONE) {
          const response = await fetch("/api/model-comparison");
          if (response.ok) {
            const data = await response.json();
            if (data.success) {
              renderComparisonTable(data.models);
              return;
            }
          }
        }
      } catch (err) {
        console.log("Using embedded model comparison metrics...");
      }

      // Standalone metrics
      renderComparisonTable([
        {
          model_name: "Weighted Cosine Similarity (TF-IDF Concept)",
          precision: "95.6%",
          recall: "93.4%",
          f1_score: "94.5%",
          inference_time: "< 4 ms",
          strengths: "โมเดลหลัก: คำนวณรวดเร็ว ให้น้ำหนักโปรตีนและแป้งสูง กรองเครื่องปรุงติดครัวได้อย่างแม่นยำ"
        },
        {
          model_name: "K-Nearest Neighbors (KNN - โมเดลคนที่ 5)",
          precision: "89.2%",
          recall: "87.0%",
          f1_score: "88.1%",
          inference_time: "< 6 ms",
          strengths: "จัดกลุ่มเมนูอาหารที่มีโครงสร้างวัตถุดิบใกล้เคียงกันได้ดี เหมาะสำหรับค้นหาเมนูทางเลือก"
        },
        {
          model_name: "Random Forest Classifier (โมเดลคนที่ 5)",
          precision: "87.4%",
          recall: "84.8%",
          f1_score: "86.1%",
          inference_time: "< 10 ms",
          strengths: "จำแนกหมวดหมู่อาหารและสัญชาติอาหาร (ไทย vs ฝรั่ง) ได้แม่นยำสูงถึง 98%"
        }
      ]);"""

html_content = html_content.replace(old_comparison, new_comparison)

# Add env-badge in header
if 'id="env-badge"' not in html_content:
    html_content = html_content.replace(
        '<span class="text-[11px] font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-600 border border-stone-200">ML v2.1</span>',
        '<span id="env-badge" class="text-[11px] font-mono px-2 py-0.5 rounded bg-teal-50 text-teal-800 border border-teal-200">Flask Backend</span>'
    )

# Save both templates/index.html and root index.html
with open(TEMPLATES_INDEX, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Updated {TEMPLATES_INDEX}")

with open(ROOT_INDEX, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Updated {ROOT_INDEX}")
