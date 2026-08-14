import os
import shutil
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Fine-Tuning Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://base.wiserly.org"


class FineTuneRequest(BaseModel):
    session_token: str
    model_id: str  # this is the model_name in the private_model table
    epochs: int = 30


@app.get("/health")
def health():
    return {"status": "ok", "worker": "YOLO Fine-Tuner"}


def fetch_user_base_model(session_token: str, model_name: str) -> str:
    """
    Calls finetune-fetch-model to pull + decrypt the user's own .pt model
    using the session token as credential. Returns the local path to the
    decrypted weights file.
    """
    res = requests.post(
        f"{SUPABASE_URL}/functions/v1/finetune-fetch-model",
        json={"session_token": session_token, "model_name": model_name},
    )

    if res.status_code != 200:
        try:
            detail = res.json().get("error", res.text)
        except Exception:
            detail = res.text
        raise Exception(f"Failed to fetch user base model: {detail}")

    local_path = f"{model_name}_base.pt"
    with open(local_path, "wb") as f:
        f.write(res.content)

    if os.path.getsize(local_path) == 0:
        raise Exception("Decrypted base model was empty")

    return local_path


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

    base_weights_path = None
    try:
        # 2. Pull + decrypt the user's own base model instead of a fixed public one
        print(f"🔐 Fetching and decrypting base model '{model_id}' for training...")
        base_weights_path = fetch_user_base_model(req.session_token, model_id)
        print(f"✅ Base model ready at {base_weights_path}")

        # 3. Run Fine-Tuning on GPU, starting from the user's own weights
        model = YOLO(base_weights_path)
        results = model.train(data="coco8.yaml", epochs=req.epochs, imgsz=640)

        weights_path = os.path.join(results.save_dir, "weights", "best.pt")
        if not os.path.exists(weights_path):
            raise Exception("Weights file not found after training completed.")

        # 4. Upload completed weights back via Edge Function
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

        if os.path.exists(results.save_dir):
            shutil.rmtree(results.save_dir)

        return {"status": "success", "message": "Fine-tuning completed and weights saved successfully!"}
    except Exception as e:
        print(f"❌ Error during training execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always scrub the decrypted base weights off Colab's disk
        if base_weights_path and os.path.exists(base_weights_path):
            os.remove(base_weights_path)
