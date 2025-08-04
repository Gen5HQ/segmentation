# sam_first_mask.py
import modal
from typing import Dict, Any
import base64, io, cv2, numpy as np
from PIL import Image

APP_NAME   = "gen5-segmentation"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
WEIGHT_LOC = "/model/sam_vit_l_0b3195.pth"

# ── 1. イメージ: CUDA 12.1 wheel を extra_index_url で取得 ─────────────
image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install(
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
        "numpy<2.0",
        "fastapi[standard]",
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
    gpu="T4",
    memory=4096,
    enable_memory_snapshot=True,
    min_containers=0,
    scaledown_window=5,
)
class SamMask:
    @modal.enter(snap=True)                 # ① CPU でロード→スナップ
    def load_cpu(self):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        self.sam = sam_model_registry["vit_l"](checkpoint=WEIGHT_LOC)
        self.generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=6,
            pred_iou_thresh=0.95,
            stability_score_thresh=0.9,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=10000,
        )
        print("loaded to cpu")

    @modal.enter(snap=False)                # ② cold-start 復元後 GPU へ
    def to_gpu(self):
        import torch
        self.sam.to("cuda" if torch.cuda.is_available() else "cpu")
        print("loaded to gpu")

    @modal.method()                         # ③ 呼び出しエンドポイント
    def get_all_masks(self, img_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        print("start")
        masks = self.generator.generate(arr)
        if not masks:
            return {"error": "No masks generated"}

        # 面積で降順ソートして上位5個まで取得
        masks_sorted = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
        
        mask_results = []
        for i, mask in enumerate(masks_sorted):
            m = mask["segmentation"].astype(np.uint8) * 255
            ok, buf = cv2.imencode(".png", m)
            if not ok:
                continue

            mask_results.append({
                "mask_base64": base64.b64encode(buf).decode(),
                "area": mask.get("area", 0),
                "stability_score": mask.get("stability_score", 0),
                "bbox": mask.get("bbox", [0, 0, 0, 0]),
            })
        print("end")
        return {
            "masks": mask_results,
            "total_masks_found": len(masks),
        }

# ── 3. HTTPSエンドポイント ─────────────────────────────
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_mask(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node.jsから呼び出し可能なHTTPSエンドポイント
    
    POSTリクエストの body:
    {
        "image": "base64エンコードされた画像データ"
    }
    
    レスポンス:
    {
        "masks": [
            {
                "mask_base64": "base64エンコードされたマスク画像",
                "area": 面積,
                "stability_score": 安定性スコア,
                "bbox": [x, y, width, height]
            },
            ...
        ],
        "total_masks_found": 見つかったマスクの総数
    }
    """
    try:
        # base64デコード
        img_base64 = item.get("image", "")
        if not img_base64:
            return {"error": "No image data provided"}
        
        img_bytes = base64.b64decode(img_base64)
        
        # SAMで処理
        print("http server loading done")
        result = SamMask().get_all_masks.remote(img_bytes)
        
        # FastAPIの場合、辞書形式で返す
        return result
        
    except Exception as e:
        return {"error": str(e)}
