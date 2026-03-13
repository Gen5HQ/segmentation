import os
import io
import base64
import time
from threading import Lock
import torch
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from contextlib import asynccontextmanager

MODEL_TYPE = "vit_l"
MODEL_PATH = "/models/sam_vit_l_0b3195.pth"  # docker-compose.ymlのボリュームマウント先に合わせる

# AMD GPU (ROCm) or NVIDIA GPU (CUDA) or CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Selected device: {DEVICE} (CUDA/ROCm available)")
elif hasattr(torch, 'hip') and torch.hip.is_available():
    DEVICE = "cuda"  # ROCmでもcudaデバイスとして扱われる
    print(f"Selected device: {DEVICE} (HIP available)")
else:
    DEVICE = "cpu"
    print(f"Selected device: {DEVICE} (No GPU available)")

predictor = None
predictor_lock = Lock()
gpu_cleanup_interval = max(0, int(os.getenv("GPU_CLEANUP_INTERVAL", "16")))
request_count = 0

class SegmentationRequest(BaseModel):
    image: str
    bboxes: Optional[List[List[float]]] = None
    multimask_output: Optional[bool] = False

class MaskResult(BaseModel):
    mask_base64: str
    area: int
    bbox: List[int]

class SegmentationResponse(BaseModel):
    masks: List[Dict[str, Any]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please download sam_vit_l_0b3195.pth")

    print(f"Loading SAM model from {MODEL_PATH}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
    sam.to(DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print(f"Model loaded successfully on {DEVICE}")

    yield

    if predictor is not None:
        predictor.reset_image()
        del predictor
        predictor = None
    if DEVICE != "cpu":
        torch.cuda.empty_cache()

app = FastAPI(title="SAM Segmentation API", lifespan=lifespan)


def _decode_image(image_base64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(img_bytes)) as img:
        rgb_img = img.convert("RGB")
        return np.array(rgb_img, dtype=np.uint8, copy=True)


def _cleanup_predictor_state() -> None:
    global request_count

    if predictor is None:
        return

    predictor.reset_image()
    if DEVICE == "cpu" or gpu_cleanup_interval == 0:
        return

    request_count += 1
    if request_count % gpu_cleanup_interval == 0:
        torch.cuda.empty_cache()


def _generate_masks(request: SegmentationRequest) -> Dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.image:
        raise HTTPException(status_code=400, detail="No image data provided")

    start_time = time.time()
    arr = _decode_image(request.image)
    img_height, img_width = arr.shape[:2]
    mask_results = []

    try:
        with torch.inference_mode():
            predictor.set_image(arr)

            if request.bboxes:
                for bbox_idx, bbox in enumerate(request.bboxes):
                    if len(bbox) != 4:
                        raise HTTPException(status_code=400, detail=f"Invalid bbox format at index {bbox_idx}")

                    x1_norm, y1_norm, x2_norm, y2_norm = bbox

                    x1 = max(0.0, min(float(x1_norm) * img_width, img_width))
                    x2 = max(0.0, min(float(x2_norm) * img_width, img_width))
                    y1 = max(0.0, min(float(y1_norm) * img_height, img_height))
                    y2 = max(0.0, min(float(y2_norm) * img_height, img_height))

                    if x2 <= x1 or y2 <= y1:
                        raise HTTPException(status_code=400, detail=f"Invalid bbox coordinates at index {bbox_idx}")

                    bbox_sam_format = np.array([x1, y1, x2, y2], dtype=np.float32)

                    masks, scores, _ = predictor.predict(
                        box=bbox_sam_format,
                        multimask_output=request.multimask_output
                    )

                    if len(masks) == 0:
                        continue

                    best_mask_idx = int(np.argmax(scores))
                    best_mask = masks[best_mask_idx]
                    best_score = float(scores[best_mask_idx])

                    mask_image = best_mask.astype(np.uint8) * 255
                    ok, buf = cv2.imencode(".png", mask_image)
                    if not ok:
                        continue

                    area = int(np.count_nonzero(best_mask))

                    mask_results.append({
                        "mask_base64": base64.b64encode(buf).decode(),
                        "area": area,
                        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                        "score": best_score
                    })
            else:
                bbox_sam_format = np.array([0, 0, img_width, img_height], dtype=np.float32)

                masks, scores, _ = predictor.predict(
                    box=bbox_sam_format,
                    multimask_output=False
                )

                if len(masks) > 0:
                    best_mask = masks[0]
                    best_score = float(scores[0])

                    mask_image = best_mask.astype(np.uint8) * 255
                    ok, buf = cv2.imencode(".png", mask_image)

                    if ok:
                        area = int(np.count_nonzero(best_mask))
                        mask_results.append({
                            "mask_base64": base64.b64encode(buf).decode(),
                            "area": area,
                            "bbox": [0, 0, img_width, img_height],
                            "score": best_score
                        })

        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.3f} seconds")

        return {"masks": mask_results}
    finally:
        _cleanup_predictor_state()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "model": MODEL_TYPE,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate_mask", response_model=SegmentationResponse)
def generate_mask(request: SegmentationRequest) -> Dict[str, Any]:
    try:
        with predictor_lock:
            return _generate_masks(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
