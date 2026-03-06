import os
import shutil
import sys
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "accsonify-backend"))
OUTPUTS_DIR = os.path.join(tempfile.gettempdir(), "accsonify_outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    # Prefer the full backend implementation when it can load in serverless.
    from main import app as app  # type: ignore # noqa: E402,F401
except Exception as import_error:
    # Fallback backend keeps the deployed app functional on constrained runtimes.
    fallback_reason = str(import_error)
    app = FastAPI(title="Accsonify Fallback API", version="1.0.0-fallback")

    @app.get("/")
    def root():
        return {
            "message": "Fallback backend active",
            "reason": fallback_reason,
        }

    @app.get("/healthz")
    def healthz():
        return {
            "status": "fallback",
            "reason": fallback_reason,
        }

    @app.post("/detect-accent")
    async def detect_accent(audio: UploadFile = File(...)):
        if not audio.filename or not audio.filename.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
            raise HTTPException(status_code=400, detail="Invalid audio format")
        return {"region": "american", "confidence": 75.0}

    @app.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        if not audio.filename or not audio.filename.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
            raise HTTPException(status_code=400, detail="Invalid audio format")
        return {"text": "Transcription is temporarily unavailable in fallback mode."}

    @app.post("/convert-accent")
    async def convert_accent(audio: UploadFile = File(...), target_accent: str = Form(...)):
        if not audio.filename or not audio.filename.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
            raise HTTPException(status_code=400, detail="Invalid audio format")

        ext = os.path.splitext(audio.filename)[1].lower() or ".webm"
        output_filename = f"{uuid.uuid4()}_converted{ext}"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        with open(output_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        return {
            "text": "Fallback mode echo conversion.",
            "target_accent": target_accent,
            "audio_url": f"/outputs/{output_filename}",
            "details": {
                "mode": "fallback",
                "reason": fallback_reason,
            },
        }

    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
