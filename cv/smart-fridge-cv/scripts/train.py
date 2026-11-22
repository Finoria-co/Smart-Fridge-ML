"""เทรน YOLO11 (detect) บน dataset foodseg_fridge_thai_yolo

รันใน WSL (ดู CLAUDE.md):
    /root/venvs/smart-fridge/bin/python scripts/train.py --name smoke --epochs 1 --fraction 0.1
    /root/venvs/smart-fridge/bin/python scripts/train.py --name baseline_n --epochs 50

เชื่อมกับบทเรียน:
- บท 05 ANN: YOLO = ANN หลายชั้น เทรนด้วย gradient descent (optimizer) + backprop, lr0 = learning rate
- บท 03 Decision Tree: ใช้ val ตรวจ overfitting, patience = early stopping (แนวคิดเดียวกับ pre-pruning)
"""
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # โฟลเดอร์ smart-fridge-cv
DEFAULT_DATA = "/root/data/foodseg_yolo/data.yaml"  # สำเนาบน ext4 ของ WSL (อ่านเร็วกว่า /mnt/c)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="ชื่อการทดลอง → runs/detect/<name>")
    p.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt / yolo11s.pt / path ของ .pt")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15, help="early stopping: หยุดถ้า val ไม่ดีขึ้นกี่ epoch")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=float, default=16, help="จำนวนเต็ม หรือ -1 ให้ Ultralytics เลือกตาม GPU memory")
    p.add_argument("--fraction", type=float, default=1.0, help="สัดส่วนของ train ที่ใช้ (ทดลองสั้นใช้ 0.1)")
    p.add_argument("--workers", type=int, default=2, help="WSL มี RAM ~7 GB อย่าเกิน 4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="false", choices=["false", "ram", "disk"],
                   help="ram เร็วสุด แต่ต้องมี RAM เหลือ ~10 GB (Colab ได้, WSL ห้าม)")
    p.add_argument("--project", default=str(ROOT / "runs" / "detect"),
                   help="โฟลเดอร์เก็บผล (Colab: ชี้ไปที่ Google Drive เพื่อไม่ให้หายเมื่อ session หลุด)")
    p.add_argument("--resume", metavar="LAST_PT",
                   help="เทรนต่อจาก checkpoint (…/weights/last.pt) ใช้ค่าเดิมทั้งหมดจาก run นั้น")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO  # import ช้า ให้ --help ตอบเร็ว

    t0 = time.time()
    if args.resume:
        # resume ใช้ args ทั้งหมดที่บันทึกใน checkpoint (epochs, batch, data, project ...)
        # ถ้า path ของ data เปลี่ยน (เช่น Colab session ใหม่) ให้แตก zip ไว้ที่เดิม
        model = YOLO(args.resume)
        model.train(resume=True)
        print(f"\n[train.py] resume: เวลารอบนี้ {(time.time() - t0) / 60:.1f} นาที → {model.trainer.save_dir}")
        return

    batch = int(args.batch) if args.batch == int(args.batch) else args.batch
    cache = False if args.cache == "false" else args.cache
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=batch,
        fraction=args.fraction,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        cache=cache,
        project=args.project,
        name=args.name,
        exist_ok=False,
        plots=True,
    )
    minutes = (time.time() - t0) / 60
    print(f"\n[train.py] {args.name}: เวลาเทรนรวม {minutes:.1f} นาที → {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
