import os
import io
import base64
import torch
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from contextlib import asynccontextmanager

MODEL_TYPE = "vit_b"
MODEL_PATH = "/models/sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = None

class SegmentationRequest(BaseModel):
    image: str
    bboxes: Optional[List[List[int]]] = None
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

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image data provided")

        img_bytes = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

        predictor.set_image(arr)

        mask_results = []

        if request.bboxes:
            for bbox_idx, bbox in enumerate(request.bboxes):
                if len(bbox) != 4:
                    raise HTTPException(status_code=400, detail=f"Invalid bbox format at index {bbox_idx}")

                x, y, w, h = bbox
                bbox_sam_format = np.array([x, y, x + w, y + h])

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
                    "bbox": bbox,
                    "score": best_score
                })
        else:
            h, w = arr.shape[:2]
            bbox_sam_format = np.array([0, 0, w, h])

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
                        "bbox": [0, 0, w, h],
                        "score": best_score
                    })

        return {"masks": mask_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)