import os
import sys
import json
import shutil
import zipfile
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# --- Mirror all stdout (including Ultralytics' own training logs) to a file. ---
LOG_FILE_PATH = "/content/worker.log"

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return self.streams[0].isatty() if hasattr(self.streams[0], "isatty") else False
    def fileno(self):
        return self.streams[0].fileno()
    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

_log_file = open(LOG_FILE_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

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

# Stock architectures Ultralytics can auto-download when starting a brand new model.
ALLOWED_BASE_ARCHITECTURES = {
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
}


class FineTuneRequest(BaseModel):
    session_token: str
    model_id: str
    project_id: str
    epochs: int = 30
    mode: str = "finetune"
    base_architecture: Optional[str] = None
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "auto"
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


class UploadRequest(BaseModel):
    session_token: str


@app.get("/health")
def health():
    return {"status": "ok", "worker": "YOLO Fine-Tuner"}


def update_status(session_token: str, status: str, error_message: str = None, **extra):
    try:
        payload = {"session_token": session_token, "status": status, **extra}
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


def resolve_base_weights(req: FineTuneRequest, model_id: str) -> str:
    if req.mode == "new":
        arch = (req.base_architecture or "").strip().lower()
        if arch not in ALLOWED_BASE_ARCHITECTURES:
            raise Exception(
                f"Invalid or missing base_architecture '{arch}' for mode='new'. "
                f"Must be one of: {sorted(ALLOWED_BASE_ARCHITECTURES)}"
            )
        print(f"🆕 Starting a NEW model '{model_id}' from stock architecture '{arch}'...")
        return f"{arch}.pt"
    else:
        print(f"🔐 Fetching and decrypting base model '{model_id}' for fine-tuning...")
        path = fetch_user_base_model(req.session_token, model_id)
        print(f"✅ Base model ready at {path}")
        return path


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


def run_finetune_job(req: FineTuneRequest, model_id: str):
    base_weights_path = None
    is_downloaded_base = False
    update_status(
        req.session_token, "training",
        project_id=req.project_id, model_id=model_id, mode=req.mode,
        base_architecture=req.base_architecture if req.mode == "new" else None,
        parent_model=model_id if req.mode == "finetune" else None,
        epochs=req.epochs,
    )
    try:
        base_weights_path = resolve_base_weights(req, model_id)
        is_downloaded_base = req.mode != "new"

        print(f"📦 Exporting dataset for project '{req.project_id}'...")
        data_yaml_path = fetch_dataset(req.session_token, req.project_id)
        print(f"✅ Dataset ready at {data_yaml_path}")

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

        with open(LAST_RUN_PATH, "w") as f:
            json.dump({
                "session_token": req.session_token,
                "model_id": model_id,
                "mode": req.mode,
                "weights_path": weights_path,
                "metrics": metrics,
                "epochs": req.epochs,
                "train_kwargs": {k: v for k, v in train_kwargs.items() if k != "data"},
            }, f)

        print("✅ Training complete. Results saved for review.")
        print(f"📊 Metrics: {metrics}")
        print("👉 You can now deploy these weights directly from your web app UI dashboard!")

        update_status(
            req.session_token, "review",
            hyperparameters={k: v for k, v in train_kwargs.items() if k != "data"},
            metrics=metrics,
        )
    except Exception as e:
        print(f"❌ Error during training execution: {str(e)}")
        update_status(req.session_token, "failed", str(e))
    finally:
        if is_downloaded_base and base_weights_path and os.path.exists(base_weights_path):
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

    if req.mode == "new":
        model_id = req.model_id
        print(f"🚀 Token verified! Starting NEW model training: '{model_id}' from '{req.base_architecture}'...")
    else:
        model_id = session_data.get("model_id", req.model_id)
        print(f"🚀 Token verified! Starting YOLOv8 fine-tuning for model: {model_id}...")

    background_tasks.add_task(run_finetune_job, req, model_id)

    return {
        "status": "started",
        "message": "Training dispatched. Check Colab logs for progress and review results in your web dashboard.",
        "model_id": model_id,
        "mode": req.mode,
    }


@app.post("/upload-weights")
def trigger_frontend_upload(req: UploadRequest):
    """Secure endpoint invoked from the web dashboard UI to upload best.pt directly to Supabase."""
    if not os.path.exists(LAST_RUN_PATH):
        raise HTTPException(status_code=404, detail="No training run results found on worker.")
    
    with open(LAST_RUN_PATH, "r") as f:
        run_data = json.load(f)
    
    if run_data.get("session_token") != req.session_token:
        raise HTTPException(status_code=403, detail="Session token mismatch.")
    
    weights_path = run_data.get("weights_path")
    if not weights_path or not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Weights file 'best.pt' not found on disk.")
    
    model_id = run_data.get("model_id")
    
    print(f"📤 Uploading weights for model '{model_id}' to Supabase via worker proxy...")
    
    try:
        with open(weights_path, "rb") as f:
            weights_bytes = f.read()
            
        res = requests.post(
            f"{SUPABASE_URL}/functions/v1/complete-finetune-session",
            headers={"Authorization": f"Bearer {req.session_token}"},
            files={"file": ("best.pt", weights_bytes, "application/octet-stream")},
            data={"model_id": model_id},
            timeout=60
        )
        
        if res.status_code != 200:
            raise Exception(f"Supabase rejected upload: {res.text}")
            
        print("✅ Weights successfully uploaded and model updated via frontend request!")
        update_status(req.session_token, "completed")
        
        return {"status": "success", "message": "Model weights successfully deployed."}
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
