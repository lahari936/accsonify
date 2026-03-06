import os
import sys

from fastapi import FastAPI, HTTPException

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "accsonify-backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    # Prefer the full backend implementation when it can load in serverless.
    from main import app as app  # type: ignore # noqa: E402,F401
except Exception as import_error:
    # Fallback backend keeps the deployed app functional on constrained runtimes.
    fallback_reason = str(import_error)
    app = FastAPI(title="Accsonify Fallback API", version="1.0.0-fallback")

    def _raise_real_inference_required():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Real ML backend is unavailable on this deployment",
                "reason": fallback_reason,
                "action": "Deploy accsonify-backend to a dedicated Python host and set NEXT_PUBLIC_API_BASE_URL to that backend URL",
            },
        )

    @app.get("/")
    def root():
        return {
            "message": "Fallback backend active",
            "reason": fallback_reason,
        }

    @app.get("/healthz")
    def healthz():
        return {
            "status": "degraded",
            "reason": fallback_reason,
        }

    @app.post("/detect-accent")
    async def detect_accent():
        _raise_real_inference_required()

    @app.post("/transcribe")
    async def transcribe():
        _raise_real_inference_required()

    @app.post("/convert-accent")
    async def convert_accent():
        _raise_real_inference_required()
