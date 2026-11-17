"""zip dataset ที่แบ่ง train/val/test แล้ว เพื่ออัปโหลดขึ้น Google Drive ใช้กับ Colab

    python scripts/zip_dataset.py <โฟลเดอร์ foodseg_yolo> <ไฟล์ zip ปลายทาง>

- ข้ามไฟล์ *.cache ของ Ultralytics (สร้างใหม่เองบน Colab)
- ใช้ ZIP_STORED เพราะรูป .jpg บีบอัดอยู่แล้ว บีบซ้ำไม่ได้ขนาดลดแต่เสียเวลา
- ใน zip มีโฟลเดอร์บนสุดชื่อ foodseg_yolo/ (แตกแล้วได้ foodseg_yolo/data.yaml)
"""
import sys
import zipfile
from collections import Counter
from pathlib import Path


def main(src: Path, dst: Path):
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix != ".cache")
    counts = Counter()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in files:
            rel = p.relative_to(src)
            zf.write(p, Path(src.name) / rel)
            if len(rel.parts) >= 2 and rel.parts[0] in ("images", "labels"):
                counts[(rel.parts[0], rel.parts[1])] += 1
    for (kind, split), n in sorted(counts.items()):
        print(f"{kind:7s} {split:6s} {n:6d}")
    print(f"{len(files)} files → {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
