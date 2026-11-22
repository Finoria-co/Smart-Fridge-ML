"""ทดสอบส่วนที่ไม่ต้องใช้โมเดล: ตาราง ingredient_map.json และ aggregate()

    python tests/test_predict.py      (หรือ pytest tests/)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict import aggregate, load_ingredient_map  # noqa: E402

CONTRACT_12 = ["pork", "chicken", "shrimp", "bacon", "egg", "rice", "pasta",
               "bread", "holy_basil", "tomato", "cheese", "cabbage"]


def model_class_names():
    text = (ROOT / "data" / "data.yaml").read_text(encoding="utf-8")
    return {m.group(1).strip() for m in re.finditer(r"^\s+\d+: (.+)$", text, re.M)}


def test_contract_has_exactly_12():
    contract, _ = load_ingredient_map()
    assert sorted(contract) == sorted(CONTRACT_12)


def test_mapped_classes_exist_in_model():
    _, class_to_std = load_ingredient_map()
    missing = set(class_to_std) - model_class_names()
    assert not missing, f"class ใน ingredient_map.json ที่ไม่มีใน data.yaml: {missing}"


def test_rice_pasta_not_detectable():
    contract, _ = load_ingredient_map()
    assert contract["rice"]["classes"] == [] and contract["pasta"]["classes"] == []


def test_aggregate_counts_and_merges():
    dets = [
        ("egg", 0.9, [0, 0, 10, 10]),
        ("egg", 0.8, [20, 0, 30, 10]),
        ("bacon", 0.7, [0, 50, 40, 90]),
        ("ham", 0.6, [1, 51, 40, 90]),     # ชิ้นเดียวกับ bacon -> ไม่นับซ้ำ
        ("ham", 0.5, [100, 100, 140, 140]),  # อีกชิ้น -> นับ
        ("carrot", 0.95, [0, 0, 5, 5]),    # นอก contract
    ]
    out = aggregate(dets)
    assert [o["ingredient"] for o in out] == ["egg", "bacon"]
    egg, bacon = out
    assert egg["count"] == 2 and egg["confidence"] == 0.9 and egg["name_th"] == "ไข่ไก่"
    assert bacon["count"] == 2 and bacon["confidence"] == 0.7

    out_all = aggregate(dets, only_contract=False)
    assert out_all[0]["ingredient"] == "carrot" and out_all[0]["in_contract"] is False
    assert out_all[0]["name_th"] is None


def test_aggregate_space_to_underscore_and_empty():
    assert aggregate([]) == []
    out = aggregate([("holy basil", 0.4, [0, 0, 1, 1]), ("spring onion", 0.3, [0, 0, 1, 1])], only_contract=False)
    assert [o["ingredient"] for o in out] == ["holy_basil", "spring_onion"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
