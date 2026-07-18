import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import torch

from model import load_model, predict

# --- 1. Initialize FastAPI App and CORS ---
app = FastAPI(title="Image Anti-Spoofing API")

origins = [
    "https://imageantispoofing.netlify.app",  # update once your frontend is live
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load Model from Hugging Face Hub ---
HF_REPO_ID = "LaabhGupta/image-antispoofing"
MODEL_FILENAME = "deeper_cnn_model.pth"  # recommended model - see README for why
ARCHITECTURE = "deeper"
device = "cpu"

print("🔍 Downloading model from Hugging Face Hub...")
weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)

print("🧠 Loading model...")
model = load_model(ARCHITECTURE, weights_path, device=device)
print("✔ Model loaded successfully (Deeper CNN)")

# --- 3. Constants ---
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]


@app.get("/")
def health():
    return {"message": "Image Anti-Spoofing API Running!"}


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"}

    try:
        file_bytes = await file.read()
        label, confidence = predict(file_bytes, model, device=device)

        return {
            "filename": file.filename,
            "predicted_class": label,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}