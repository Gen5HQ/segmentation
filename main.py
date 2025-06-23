# sam_first_mask.py
import modal
from typing import Dict, Any
import base64, io, cv2, numpy as np
from PIL import Image

APP_NAME   = "sam-first-mask-fast"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
WEIGHT_LOC = "/model/sam_vit_h_4b8939.pth"

# ── 1. イメージ: CUDA 12.1 wheel を extra_index_url で取得 ─────────────
image = (
    modal.Image.debian_slim()
    .apt_install(                       # ← curl を先頭に追加
        "curl",
        "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    )
    .pip_install(
        "torch==2.2.1+cu121",
        "torchvision==0.17.1+cu121",
        "segment-anything",
        "opencv-python-headless",
        "Pillow",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        [
            # curl が使えるようになる
            "mkdir -p /model",
            f"curl -L -o {WEIGHT_LOC} {WEIGHT_URL}",
            f"echo '✓ SAM weight saved to {WEIGHT_LOC}'",
        ]
    )
)

# ── 2. SAM マスク生成クラス ─────────────────────────────
app = modal.App(APP_NAME)

@app.cls(
    image=image,
    gpu="A10G",
    memory=4096,
    enable_memory_snapshot=True,
    min_containers=0,
    scaledown_window=5,
)
class SamMask:
    @modal.enter(snap=True)                 # ① CPU でロード→スナップ
    def load_cpu(self):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        self.sam = sam_model_registry["vit_h"](checkpoint=WEIGHT_LOC)
        self.generator = SamAutomaticMaskGenerator(self.sam)

    @modal.enter(snap=False)                # ② cold-start 復元後 GPU へ
    def to_gpu(self):
        import torch
        self.sam.to("cuda" if torch.cuda.is_available() else "cpu")

    @modal.method()                         # ③ 呼び出しエンドポイント
    def get_first_mask(self, img_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        masks = self.generator.generate(arr)
        if not masks:
            return {"error": "No masks generated"}

        m0 = masks[0]["segmentation"].astype(np.uint8) * 255
        ok, buf = cv2.imencode(".png", m0)
        if not ok:
            return {"error": "Encode failed"}

        return {
            "mask_base64": base64.b64encode(buf).decode(),
            "total_masks": len(masks),
            "mask_area": masks[0].get("area", 0),
            "stability_score": masks[0].get("stability_score", 0),
        }

# ── 3. ローカルテスト ───────────────────────────────────
@app.local_entrypoint()
def main(path: str = "sample.jpg"):
    with open(path, "rb") as f:
        bytes_ = f.read()
    res = SamMask().get_first_mask.remote(bytes_)
    print(res)
