# sam_first_mask.py
import modal
from typing import Dict, Any

APP_NAME   = "gen5-segmentation"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
WEIGHT_LOC = "/model/sam_vit_l_0b3195.pth"

# ── 1. イメージ: CUDA 12.1 wheel を extra_index_url で取得 ─────────────
image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install(
        "curl",
        "libgl1-mesa-dev", "libglib2.0-0",
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
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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
        self.predictor = SamPredictor(self.sam)
        # セッション管理用の辞書
        self.sessions = {}
        print("loaded to cpu")

    @modal.enter(snap=False)                # ② cold-start 復元後 GPU へ
    def to_gpu(self):
        import torch
        self.sam.to("cuda" if torch.cuda.is_available() else "cpu")
        print("loaded to gpu")

    @modal.method()                         # ③ 新しいbbox対応エンドポイント
    def get_masks_for_bboxes(self, img_bytes: bytes, bboxes: list, multimask_output: bool = False) -> Dict[str, Any]:
        """
        指定されたbboxesに対して、それぞれ最も近い1つのマスクのみを返す
        
        Args:
            img_bytes: 画像データ
            bboxes: [[x, y, width, height], ...] 形式のbbox座標リスト
            multimask_output: 複数マスク出力を有効にするか（今回は使用しない）
        """
        import torch, base64, io, cv2, numpy as np
        from PIL import Image
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        
        # 画像を設定
        self.predictor.set_image(arr)
        
        mask_results = []
        
        for bbox_idx, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            # SAMではbboxは[x_min, y_min, x_max, y_max]形式
            bbox_sam_format = np.array([x, y, x + w, y + h])
            
            # マスク予測を実行
            masks, scores, _ = self.predictor.predict(
                box=bbox_sam_format,
                multimask_output=multimask_output
            )
            
            if len(masks) == 0:
                continue
                
            # 最もスコアが高いマスクを選択
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            # マスクをPNG形式にエンコード
            m = best_mask.astype(np.uint8) * 255
            ok, buf = cv2.imencode(".png", m)
            if not ok:
                continue
            
            # マスクの面積を計算
            area = int(np.sum(best_mask))
            
            mask_results.append({
                "mask_base64": base64.b64encode(buf).decode(),
                "area": area,
                "bbox": bbox,
            })
        
        return {"masks": mask_results}


    @modal.method()
    def init_session(self, session_id: str, img_bytes: bytes) -> Dict[str, Any]:
        """
        セッションを初期化し、画像を設定する
        """
        import io, numpy as np
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)

            # predictorに画像を設定
            self.predictor.set_image(arr)

            # セッション情報を保存
            self.sessions[session_id] = {
                "has_image": True,
                "curr_lowres": None,  # 前回のlow-res logits
                "image_shape": arr.shape[:2]  # (H, W)
            }

            return {
                "session_id": session_id,
                "status": "initialized",
                "image_shape": arr.shape[:2]
            }
        except Exception as e:
            return {"error": f"Failed to initialize session: {str(e)}"}

    @modal.method()
    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        セッションベースの推論を実行
        """
        import numpy as np, base64, io, cv2

        session_id = payload.get("session_id")
        if not session_id or session_id not in self.sessions:
            return {"error": "Invalid or missing session_id"}

        session = self.sessions[session_id]
        if not session.get("has_image"):
            return {"error": "Session not initialized with image"}

        mode = payload.get("mode")

        if mode == "points":
            pts = payload.get("points", [])
            if not pts:
                return {"error": "No points provided"}

            # point_coords と point_labels を構築
            point_coords = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
            point_labels = np.array([p["label"] for p in pts], dtype=np.int32)

            # オプションのbox
            box = payload.get("box")
            box_np = np.array(box, dtype=np.float32) if box else None

            # 前回のlow-res logitsを取得
            mask_input = session.get("curr_lowres")

            # 推論実行
            masks, scores, lowres = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_np,
                mask_input=mask_input,
                multimask_output=payload.get("multimask_output", True)
            )

            if len(masks) == 0:
                return {"error": "No masks generated"}

            # 最高スコアのマスクを選択
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            # 次回用にlow-res logitsを保存
            session["curr_lowres"] = lowres[best_idx][None, ...]

            # マスクをPNG形式にエンコード
            mask_uint8 = best_mask.astype(np.uint8) * 255
            ok, buf = cv2.imencode(".png", mask_uint8)
            if not ok:
                return {"error": "Failed to encode mask"}

            mask_base64 = base64.b64encode(buf).decode()
            area = int(np.sum(best_mask))

            return {
                "mask": mask_base64,
                "score": best_score,
                "area": area,
                "candidates": len(scores),
                "session_id": session_id
            }

        else:
            return {"error": f"Unsupported mode: {mode}"}

# ── 3. HTTPSエンドポイント ─────────────────────────────
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_mask(item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # base64デコード
        img_base64 = item.get("image", "")
        if not img_base64:
            return {"error": "No image data provided"}
        
        import base64
        img_bytes = base64.b64decode(img_base64)
        
        # pointsまたはbboxesの処理
        points = item.get("points")
        bboxes = item.get("bboxes")
        multimask_output = item.get("multimask_output", False)

        if points is not None:
            # points指定時はセッションベースを使用
            session_id = item.get("session_id", "temp_session")
            sam_instance = SamMask()

            # 一時的にセッション初期化
            init_result = sam_instance.init_session.remote(session_id, img_bytes)
            if "error" in init_result:
                return init_result

            # points推論
            payload = {
                "session_id": session_id,
                "mode": "points",
                "points": points,
                "multimask_output": multimask_output
            }
            result = sam_instance.infer.remote(payload)

        elif bboxes is not None:
            # bbox指定時の処理
            print("http server loading done (bbox mode)")
            result = SamMask().get_masks_for_bboxes.remote(img_bytes, bboxes, multimask_output)
        else:
            # pointsもbboxesも指定されていない場合はエラー
            return {"error": "Either 'points' or 'bboxes' parameter is required"}

        # FastAPIの場合、辞書形式で返す
        return result

    except Exception as e:
        return {"error": str(e)}
