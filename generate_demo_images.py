import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("static/demo_samples", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

def create_fridge_sample(filename, title, subtitle, items_info, bg_color):
    width, height = 700, 480
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # วาดชั้นวางของในตู้เย็น (Shelves)
    draw.rectangle([20, 20, width - 20, height - 20], outline="#cbd5e1", width=3)
    draw.line([20, 240, width - 20, 240], fill="#94a3b8", width=4)
    draw.line([20, 360, width - 20, 360], fill="#94a3b8", width=4)

    # หัวข้อ
    draw.text((35, 30), f"FRIDGE PHOTO: {title}", fill="#ffffff")
    draw.text((35, 52), f"Scenario: {subtitle}", fill="#94a3b8")

    # วาดกล่องวัตถุดิบจำลอง
    for item in items_info:
        x1, y1, x2, y2 = item["rect"]
        color = item["color"]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=color, outline="#ffffff", width=2)
        # ป้ายชื่อวัตถุดิบ
        label = item["name"]
        draw.text((x1 + 10, y1 + 12), label, fill="#ffffff")
        draw.text((x1 + 10, y1 + 32), f"({item['en']})", fill="#e2e8f0")

    save_path = os.path.join("static", "demo_samples", filename)
    img.save(save_path, "JPEG", quality=92)
    print(f"Created: {save_path}")

# 1. ตู้เย็นไทย: ผัดกะเพราหมูสับไข่ดาว
create_fridge_sample(
    "fridge_thai_basil.jpg",
    "Thai Cuisine Ingredients",
    "Minced Pork, Fresh Eggs & Holy Basil",
    [
        {"rect": [60, 100, 240, 210], "color": "#be123c", "name": "หมูสับสด", "en": "Minced Pork"},
        {"rect": [280, 90, 440, 215], "color": "#d97706", "name": "ไข่ไก่สด (3 ฟอง)", "en": "Fresh Eggs"},
        {"rect": [480, 100, 640, 210], "color": "#047857", "name": "ใบกะเพราสด", "en": "Holy Basil"},
        {"rect": [60, 260, 220, 340], "color": "#475569", "name": "กระเทียม/พริก", "en": "Chili & Garlic"},
        {"rect": [280, 260, 480, 340], "color": "#0284c7", "name": "ข้าวสวยแช่เย็น", "en": "Cooked Rice"}
    ],
    bg_color="#0f172a"
)

# 2. ตู้เย็นฝรั่ง: พาสต้าคาโบนาร่า
create_fridge_sample(
    "fridge_western_carbonara.jpg",
    "Western Pasta Ingredients",
    "Pasta, Smoked Bacon, Cheese & Eggs",
    [
        {"rect": [60, 95, 230, 215], "color": "#ca8a04", "name": "เส้นพาสต้า", "en": "Pasta Spaghet"},
        {"rect": [260, 95, 430, 215], "color": "#991b1b", "name": "เบคอนรมควัน", "en": "Smoked Bacon"},
        {"rect": [460, 95, 640, 215], "color": "#eab308", "name": "ชีสแท้", "en": "Parmesan Cheese"},
        {"rect": [150, 260, 320, 345], "color": "#d97706", "name": "ไข่ไก่สด", "en": "Fresh Eggs"},
        {"rect": [360, 260, 550, 345], "color": "#2563eb", "name": "นมสด/ครีม", "en": "Fresh Milk"}
    ],
    bg_color="#18181b"
)

# 3. ตู้เย็นบรันช์: แซนด์วิช/ออมเล็ต
create_fridge_sample(
    "fridge_brunch_sandwich.jpg",
    "Western Brunch Ingredients",
    "Bread, Bacon, Cheese & Tomatoes",
    [
        {"rect": [60, 95, 240, 215], "color": "#b45309", "name": "ขนมปังแผ่น", "en": "Bread Loaf"},
        {"rect": [270, 95, 440, 215], "color": "#eab308", "name": "เชดด้าชีส", "en": "Cheddar Cheese"},
        {"rect": [470, 95, 640, 215], "color": "#dc2626", "name": "มะเขือเทศสด", "en": "Fresh Tomatoes"},
        {"rect": [140, 260, 330, 345], "color": "#991b1b", "name": "เบคอนแผ่น", "en": "Bacon Slices"},
        {"rect": [370, 260, 540, 345], "color": "#f59e0b", "name": "เนยสด", "en": "Fresh Butter"}
    ],
    bg_color="#1e1e24"
)
