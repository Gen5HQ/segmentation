import os
import io
import base64
import time
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
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please download sam_vit_b_01ec64.pth")

    print(f"Loading SAM model from {MODEL_PATH}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    print(f"Model loaded successfully on {DEVICE}")

    yield

    del predictor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(title="SAM Segmentation API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "model": MODEL_TYPE,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate_mask", response_model=SegmentationResponse)
async def generate_mask(request: SegmentationRequest) -> Dict[str, Any]:
    global predictor

    start_time = time.time()

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image data provided")

        img_bytes = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

        img_height, img_width = arr.shape[:2]

        predictor.set_image(arr)

        mask_results = []

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

                bbox_sam_format = np.array([x1, y1, x2, y2])

                masks, scores, _ = predictor.predict(
                    box=bbox_sam_format,
                    multimask_output=request.multimask_output
                )

                if len(masks) == 0:
                    continue

                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                best_score = float(scores[best_mask_idx])

                m = best_mask.astype(np.uint8) * 255
                ok, buf = cv2.imencode(".png", m)
                if not ok:
                    continue

                area = int(np.sum(best_mask))

                mask_results.append({
                    "mask_base64": base64.b64encode(buf).decode(),
                    "area": area,
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "score": best_score
                })
        else:
            bbox_sam_format = np.array([0, 0, img_width, img_height])

            masks, scores, _ = predictor.predict(
                box=bbox_sam_format,
                multimask_output=False
            )

            if len(masks) > 0:
                best_mask = masks[0]
                best_score = float(scores[0])

                m = best_mask.astype(np.uint8) * 255
                ok, buf = cv2.imencode(".png", m)

                if ok:
                    area = int(np.sum(best_mask))
                    mask_results.append({
                        "mask_base64": base64.b64encode(buf).decode(),
                        "area": area,
                        "bbox": [0, 0, img_width, img_height],
                        "score": best_score
                    })

        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.3f} seconds")

        return {"masks": mask_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
