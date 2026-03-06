from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import tempfile
import uuid

# Import our ML logic
from model_utils import model_manager, convert_accent_tts

app = FastAPI(title="Accsonify API", version="1.0.0")

# Serverless platforms allow writes only in the system temp directory.
OUTPUTS_DIR = os.path.join(tempfile.gettempdir(), "accsonify_outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_LOADED = False


def ensure_models_loaded():
    global MODELS_LOADED
    if MODELS_LOADED:
        return

    try:
        model_manager.load_models()
        MODELS_LOADED = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to Accsonify API"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/detect-accent")
async def detect_accent(audio: UploadFile = File(...)):
    ensure_models_loaded()
    if not audio.filename.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
         raise HTTPException(status_code=400, detail="Invalid audio format")
         
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    try:
        region, confidence = model_manager.predict_accent(temp_file_path)
        # Map region to accent name
        from model_utils import region_to_accent
        accent = region_to_accent.get(region, "american")
        return {"region": accent, "confidence": round(confidence * 100, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    ensure_models_loaded()
    if not audio.filename.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
         raise HTTPException(status_code=400, detail="Invalid audio format")
         
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    try:
        text = model_manager.transcribe(temp_file_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/convert-accent")
async def convert_accent(
    audio: UploadFile = File(...), 
    target_accent: str = Form(...)
):
    ensure_models_loaded()
    if not audio.filename.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
         raise HTTPException(status_code=400, detail="Invalid audio format")
         
    temp_dir = tempfile.gettempdir()
    temp_id = uuid.uuid4()
    temp_input_path = os.path.join(temp_dir, f"{temp_id}_{audio.filename}")
    output_filename = f"{temp_id}_converted.mp3"
    temp_output_path = os.path.join(temp_dir, output_filename)
    
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    try:
        # First, transcribe to get the text
        text = model_manager.transcribe(temp_input_path)
        
        # Then, convert accent using TTS
        result = await convert_accent_tts(text, temp_input_path, target_accent, temp_output_path)
        
        # In a real app we might return the audio file directly via FileResponse
        # For ease of testing over API, we can move it to a public folder or return base64
        # Since this is local dev, let's keep it simple and return the path for now
        # OR we can serve it by configuring a static directory
        
        # Let's move the file to a static 'outputs' directory so frontend can access via URL
        final_output_path = os.path.join(OUTPUTS_DIR, output_filename)
        shutil.move(temp_output_path, final_output_path)
        
        return {
            "text": text,
            "target_accent": target_accent,
            "audio_url": f"/outputs/{output_filename}",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

# Serve static files for converted audio
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
