# Split report: foodseg_fridge_thai_yolo

สร้างด้วย `scripts/make_test_split.py` · seed 42 · test = 40% ของ val เดิม · train ไม่ถูกแตะ

| split | รูป | % | รูปว่าง |
|---|---|---|---|
| train | 10,055 | 75.4% | 1,107 |
| val | 1,959 | 14.7% | 266 |
| test | 1,317 | 9.9% | 178 |

## รูปต่อ class (train / val / test) · ⚠️ = มีใน val เดิมแต่ val หรือ test เหลือ 0

| id | class | train | val | test |
|---|---|---|---|---|
| 0 | cheese butter | 604 | 161 | 107 |
| 1 | milk | 377 | 102 | 68 |
| 2 | egg | 657 | 151 | 100 |
| 3 | apple | 303 | 78 | 52 |
| 4 | avocado | 64 | 16 | 10 |
| 5 | banana | 244 | 73 | 49 |
| 6 | strawberry | 463 | 119 | 80 |
| 7 | mango | 44 | 7 | 5 |
| 8 | lemon | 604 | 152 | 101 |
| 9 | grape | 137 | 23 | 16 |
| 10 | orange | 307 | 68 | 45 |
| 11 | steak | 749 | 209 | 139 |
| 12 | pork | 474 | 117 | 78 |
| 13 | chicken duck | 913 | 254 | 170 |
| 14 | sausage | 199 | 32 | 22 |
| 15 | fish | 259 | 58 | 38 |
| 16 | shellfish | 30 | 8 | 5 |
| 17 | shrimp | 116 | 31 | 21 |
| 18 | bread | 1074 | 275 | 184 |
| 19 | corn | 360 | 94 | 63 |
| 20 | tofu | 60 | 19 | 12 |
| 21 | eggplant | 39 | 13 | 8 |
| 22 | potato | 830 | 199 | 133 |
| 23 | garlic | 73 | 12 | 8 |
| 24 | cauliflower | 188 | 46 | 31 |
| 25 | tomato | 1048 | 285 | 190 |
| 26 | spring onion | 136 | 34 | 23 |
| 27 | rape | 54 | 13 | 8 |
| 28 | ginger | 26 | 5 | 4 |
| 29 | okra | 15 | 1 | 1 |
| 30 | lettuce | 453 | 116 | 78 |
| 31 | pumpkin | 72 | 10 | 6 |
| 32 | cucumber | 382 | 105 | 70 |
| 33 | white radish | 36 | 10 | 6 |
| 34 | carrot | 1052 | 284 | 189 |
| 35 | asparagus | 199 | 49 | 33 |
| 36 | broccoli | 788 | 210 | 140 |
| 37 | celery stick | 164 | 39 | 26 |
| 38 | cilantro mint | 1248 | 174 | 116 |
| 39 | snow peas | 48 | 14 | 10 |
| 40 | cabbage | 106 | 20 | 13 |
| 41 | bean sprouts | 24 | 8 | 5 |
| 42 | onion | 429 | 107 | 71 |
| 43 | pepper | 430 | 115 | 77 |
| 44 | green beans | 194 | 51 | 34 |
| 45 | french beans | 253 | 73 | 48 |
| 46 | mushroom | 263 | 69 | 46 |
| 47 | beans | 110 | 27 | 18 |
| 48 | blueberries | 37 | 5 | 4 |
| 49 | brussel sprouts ⚠️ | 10 | 1 | 0 |
| 50 | melon | 37 | 12 | 8 |
| 51 | meat | 145 | 37 | 25 |
| 52 | peach | 6 | 2 | 1 |
| 53 | plum | 2 | 0 | 0 |
| 54 | raspberries | 2 | 0 | 0 |
| 55 | salad | 21 | 4 | 3 |
| 56 | sauce | 65 | 17 | 11 |
| 57 | spinach | 40 | 7 | 5 |
| 58 | squash | 10 | 1 | 1 |
| 59 | watermelon | 19 | 7 | 4 |
| 60 | yogurt | 201 | 55 | 37 |
| 61 | zucchini | 13 | 2 | 2 |
| 62 | bacon | 18 | 4 | 2 |
| 63 | cream | 30 | 7 | 5 |
| 64 | cream cheese | 35 | 10 | 7 |
| 65 | flour | 16 | 6 | 4 |
| 66 | ham | 32 | 7 | 5 |
| 67 | lime | 40 | 8 | 5 |
| 68 | olive | 39 | 13 | 9 |
| 69 | sugar | 17 | 4 | 3 |
| 70 | sweet potato ⚠️ | 16 | 1 | 0 |
| 71 | sweet basil | 534 | 14 | 9 |
| 72 | kaffir lime | 624 | 14 | 9 |
| 73 | holy basil | 612 | 15 | 10 |
| 74 | wild betel | 324 | 44 | 30 |
