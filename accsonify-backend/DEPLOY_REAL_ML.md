# Deploy Real ML Backend (No Fallback)

Use this when you need real accent inference and transcription (Whisper + SVM), not mock/fallback responses.

## 1. Required model artifacts
Place these files in `accsonify-backend/` before deployment:

- `svm_model.pkl`
- `label_encoder.pkl`

If missing, API startup will fail by design (`ALLOW_MOCK_MODE=false`) to prevent wrong predictions.

## 2. Deploy backend (Render/Railway)
Create a new **Web Service** from `accsonify-backend/` using Docker.

- Runtime: Docker
- Dockerfile path: `accsonify-backend/Dockerfile`
- Health check path: `/healthz`

Environment variables:

- `ALLOW_MOCK_MODE=false`
- `FRONTEND_ORIGIN=https://<your-vercel-domain>`

## 3. Point Vercel frontend to backend
In Vercel Project Settings -> Environment Variables:

- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-domain>`

Redeploy frontend.

## 4. Verify
- `GET https://<backend-domain>/healthz` should return `{"status":"ok"}`
- Run one sample through `/detect-accent` and ensure non-random output

## 5. Train models if needed
If `svm_model.pkl` and `label_encoder.pkl` do not exist, run training locally with your dataset and copy artifacts into `accsonify-backend/` before deploy.
