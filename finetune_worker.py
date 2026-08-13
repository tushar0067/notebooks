import os
import shutil
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Fine-Tuning Worker")

# Replace with your production Supabase URL
SUPABASE_URL = "https://base.wiserly.org"

class FineTuneRequest(BaseModel):
    session_token: str
    model_id: str
    epochs: int = 30

@app.get("/health")
def health():
    return {"status": "ok", "worker": "YOLO Fine-Tuner"}

@app.post("/start-finetune")
def start_finetune(req: FineTuneRequest):
    # 1. Verify single-use token with Supabase Edge Function
    verify_res = requests.post(
        f"{SUPABASE_URL}/functions/v1/verify-finetune-session",
        json={"session_token": req.session_token}
    )
    
    if verify_res.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    session_data = verify_res.json()
    model_id = session_data.get("model_id", req.model_id)

    print(f"🚀 Token verified! Starting YOLOv8 training for model: {model_id}...")

    try:
        # 2. Run Fine-Tuning on GPU
        model = YOLO("yolov8n.pt")
        results = model.train(data="coco8.yaml", epochs=req.epochs, imgsz=640)

        # Path to best weights
        weights_path = os.path.join(results.save_dir, "weights", "best.pt")

        if not os.path.exists(weights_path):
            raise Exception("Weights file not found after training completed.")

        # 3. Upload completed weights back via Edge Function
        with open(weights_path, "rb") as f:
            weights_bytes = f.read()

        complete_res = requests.post(
            f"{SUPABASE_URL}/functions/v1/complete-finetune-session",
            headers={"Authorization": f"Bearer {req.session_token}"},
            files={"file": ("best.pt", weights_bytes, "application/octet-stream")},
            data={"model_id": model_id}
        )

        if complete_res.status_code != 200:
            raise Exception(f"Failed to upload weights: {complete_res.text}")

        # Clean up disk
        if os.path.exists(results.save_dir):
            shutil.rmtree(results.save_dir)

        return {"status": "success", "message": "Fine-tuning completed and weights saved successfully!"}

    except Exception as e:
        print(f"❌ Error during training execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
