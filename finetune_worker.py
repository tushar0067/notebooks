import os
import shutil
import zipfile
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    project_id: str  # which project's annotated dataset to train on
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


def fetch_dataset(session_token: str, project_id: str) -> str:
    """
    Calls finetune-export-dataset to get a zipped YOLO dataset (images/, labels/,
    data.yaml with the project's real classes). Extracts it to /content/dataset
    and returns the path to data.yaml.
    """
    res = requests.post(
        f"{SUPABASE_URL}/functions/v1/finetune-export-dataset",
        json={"session_token": session_token, "project_id": project_id},
    )

    if res.status_code != 200:
        try:
            detail = res.json().get("error", res.text)
        except Exception:
            detail = res.text
        raise Exception(f"Failed to export dataset: {detail}")

    dataset_dir = "/content/dataset"
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    zip_path = "/content/dataset.zip"
    with open(zip_path, "wb") as f:
        f.write(res.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    os.remove(zip_path)

    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        raise Exception("data.yaml missing from exported dataset")

    return data_yaml_path


def run_finetune_job(req: "FineTuneRequest", model_id: str):
    """The actual training work — runs in a background thread so the HTTP
    response can return immediately and the Cloudflare tunnel never times out."""
    base_weights_path = None
    try:
        # 2. Pull + decrypt the user's own base model instead of a fixed public one
        print(f"🔐 Fetching and decrypting base model '{model_id}' for training...")
        base_weights_path = fetch_user_base_model(req.session_token, model_id)
        print(f"✅ Base model ready at {base_weights_path}")

        # 3. Pull the project's real annotated dataset (replaces coco8.yaml)
        print(f"📦 Exporting dataset for project '{req.project_id}'...")
        data_yaml_path = fetch_dataset(req.session_token, req.project_id)
        print(f"✅ Dataset ready at {data_yaml_path}")

        # 4. Run Fine-Tuning on GPU, starting from the user's own weights and real data
        model = YOLO(base_weights_path)
        results = model.train(data=data_yaml_path, epochs=req.epochs, imgsz=640)

        weights_path = os.path.join(results.save_dir, "weights", "best.pt")
        if not os.path.exists(weights_path):
            raise Exception("Weights file not found after training completed.")

        # 5. Upload completed weights back via Edge Function
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

        print("✅ Fine-tuning completed and weights saved successfully!")
    except Exception as e:
        # Can't raise HTTPException here — this runs after the response was
        # already sent. Just log loudly; the frontend already got its "started" ack.
        print(f"❌ Error during training execution: {str(e)}")
    finally:
        # Always scrub the decrypted base weights off Colab's disk
        if base_weights_path and os.path.exists(base_weights_path):
            os.remove(base_weights_path)


@app.post("/start-finetune")
def start_finetune(req: FineTuneRequest, background_tasks: BackgroundTasks):
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

    # 2. Kick off the actual training in the background and respond immediately —
    #    training can run for minutes and the Cloudflare tunnel will 524 long
    #    before that if we make the caller wait for it.
    background_tasks.add_task(run_finetune_job, req, model_id)

    return {
        "status": "started",
        "message": "Training dispatched. Weights will be uploaded automatically when it finishes.",
        "model_id": model_id,
    }
