import os
import json
import shutil
import zipfile
import requests
from typing import Optional
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
LAST_RUN_PATH = "/content/last_run.json"


class FineTuneRequest(BaseModel):
    session_token: str
    model_id: str          # model_name in private_model table
    project_id: str        # which project's annotated dataset to train on
    epochs: int = 30

    # --- Real hyperparameters, all optional. Omit any of these from the
    # request and Ultralytics' own default is used — nothing is silently
    # forced except epochs/imgsz/batch/optimizer as before. ---
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "auto"          # auto, SGD, Adam, AdamW, etc.
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_epochs: Optional[float] = None
    warmup_momentum: Optional[float] = None
    warmup_bias_lr: Optional[float] = None
    box: Optional[float] = None
    cls: Optional[float] = None
    dfl: Optional[float] = None
    hsv_h: Optional[float] = None
    hsv_s: Optional[float] = None
    hsv_v: Optional[float] = None
    degrees: Optional[float] = None
    translate: Optional[float] = None
    scale: Optional[float] = None
    shear: Optional[float] = None
    perspective: Optional[float] = None
    flipud: Optional[float] = None
    fliplr: Optional[float] = None
    mosaic: Optional[float] = None
    mixup: Optional[float] = None
    copy_paste: Optional[float] = None
    patience: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok", "worker": "YOLO Fine-Tuner"}


def update_status(session_token: str, status: str, error_message: str = None):
    """Best-effort status ping — never let a status-update failure kill training."""
    try:
        payload = {"session_token": session_token, "status": status}
        if error_message:
            payload["error_message"] = error_message[:500]
        requests.post(f"{SUPABASE_URL}/functions/v1/finetune-update-status", json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Failed to update status: {e}")


def fetch_user_base_model(session_token: str, model_name: str) -> str:
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
    """Trains and saves results locally. Does NOT upload — that's a separate,
    explicit step (Cell 3) so you can review metrics before committing the
    new weights over the old model."""
    base_weights_path = None
    update_status(req.session_token, "training")
    try:
        print(f"🔐 Fetching and decrypting base model '{model_id}' for training...")
        base_weights_path = fetch_user_base_model(req.session_token, model_id)
        print(f"✅ Base model ready at {base_weights_path}")

        print(f"📦 Exporting dataset for project '{req.project_id}'...")
        data_yaml_path = fetch_dataset(req.session_token, req.project_id)
        print(f"✅ Dataset ready at {data_yaml_path}")

        # Build train() kwargs — only include hyperparams that were actually set,
        # so anything left as None falls through to Ultralytics' own default.
        train_kwargs = {
            "data": data_yaml_path,
            "epochs": req.epochs,
            "imgsz": req.imgsz,
            "batch": req.batch,
            "optimizer": req.optimizer,
        }
        optional_fields = [
            "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl",
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
            "copy_paste", "patience",
        ]
        for field in optional_fields:
            val = getattr(req, field)
            if val is not None:
                train_kwargs[field] = val

        print(f"🚀 Training with: {train_kwargs}")
        model = YOLO(base_weights_path)
        results = model.train(**train_kwargs)

        weights_path = os.path.join(results.save_dir, "weights", "best.pt")
        if not os.path.exists(weights_path):
            raise Exception("Weights file not found after training completed.")

        # Pull final metrics for the review step
        metrics = {}
        try:
            rd = results.results_dict
            metrics = {
                "mAP50": rd.get("metrics/mAP50(B)"),
                "mAP50-95": rd.get("metrics/mAP50-95(B)"),
                "precision": rd.get("metrics/precision(B)"),
                "recall": rd.get("metrics/recall(B)"),
            }
        except Exception:
            pass

        # Write everything Cell 3 needs to upload, without re-running anything
        with open(LAST_RUN_PATH, "w") as f:
            json.dump({
                "session_token": req.session_token,
                "model_id": model_id,
                "weights_path": weights_path,
                "metrics": metrics,
                "epochs": req.epochs,
                "train_kwargs": {k: v for k, v in train_kwargs.items() if k != "data"},
            }, f)

        print("✅ Training complete. Results saved for review.")
        print(f"📊 Metrics: {metrics}")
        print("👉 Happy with these results? Run Cell 3 to upload the new weights.")
        print("   Not happy? Just re-run Cell 2 with different settings — nothing was uploaded.")

        update_status(req.session_token, "review")
    except Exception as e:
        print(f"❌ Error during training execution: {str(e)}")
        update_status(req.session_token, "failed", str(e))
    finally:
        if base_weights_path and os.path.exists(base_weights_path):
            os.remove(base_weights_path)


@app.post("/start-finetune")
def start_finetune(req: FineTuneRequest, background_tasks: BackgroundTasks):
    verify_res = requests.post(
        f"{SUPABASE_URL}/functions/v1/verify-finetune-session",
        json={"session_token": req.session_token}
    )
    if verify_res.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    session_data = verify_res.json()
    model_id = session_data.get("model_id", req.model_id)
    print(f"🚀 Token verified! Starting YOLOv8 training for model: {model_id}...")

    background_tasks.add_task(run_finetune_job, req, model_id)

    return {
        "status": "started",
        "message": "Training dispatched. Check Colab logs for progress and review results before uploading.",
        "model_id": model_id,
    }
