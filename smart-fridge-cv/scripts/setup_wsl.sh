#!/usr/bin/env bash
# ตั้งค่าเครื่องสำหรับเทรน YOLO ใน WSL2 Ubuntu
# เหตุผล: Smart App Control ของ Windows บล็อก torch\lib\torch.dll (ดู experiments.md / PLAN.md Phase A)
# รัน: wsl -d Ubuntu -- bash scripts/setup_wsl.sh   (จากโฟลเดอร์ smart-fridge-cv)
set -euo pipefail

VENV="$HOME/venvs/smart-fridge"

# 1) uv แบบ user-level (ไม่ต้องใช้ sudo)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) venv บน ext4 ของ WSL (เร็วกว่า /mnt/c)
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 3) PyTorch CUDA ก่อน แล้วค่อย ultralytics (กัน pip ดึง torch แบบ CPU)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# 4) ตรวจ GPU
python - <<'EOF'
import torch, ultralytics, cv2
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
print("ultralytics", ultralytics.__version__, "cv2", cv2.__version__)
EOF
