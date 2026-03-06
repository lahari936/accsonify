# Deploy Real ML Backend (No Fallback)

Use this when you need real accent inference and transcription (Whisper + SVM), not mock/fallback responses.

## 1. Required model artifacts
Place these files in `accsonify-backend/` before deployment:

- `svm_model.pkl`
- `label_encoder.pkl`

If missing, API startup will fail by design (`ALLOW_MOCK_MODE=false`) to prevent wrong predictions.

## 2. Quick local validation with Docker
Run these from repository root (`Implementation`):

```bash
docker build -t accsonify-backend-ml ./accsonify-backend
docker run --rm -p 8000:8000 \
	-e ALLOW_MOCK_MODE=false \
	-e FRONTEND_ORIGIN=http://localhost:3000 \
	accsonify-backend-ml
```

Then test:

```bash
curl http://127.0.0.1:8000/healthz
```

If models/dependencies are correct, you should get `{"status":"ok"}`.

## 3. Deploy backend on Render (real inference)
Use one of these options:

- Option A: `New +` -> `Blueprint` -> select this repository and use `render.yaml` from repo root.
- Option B: `New +` -> `Web Service` -> Docker with:
	- Root Directory: `accsonify-backend`
	- Dockerfile Path: `accsonify-backend/Dockerfile`
	- Health Check Path: `/healthz`

Required environment variables:

- `ALLOW_MOCK_MODE=false`
- `FRONTEND_ORIGIN=https://<your-vercel-domain>`

## 4. Point Vercel frontend to backend
In Vercel Project Settings -> Environment Variables:

- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-domain>`

Redeploy frontend.

## 5. Verify
- `GET https://<backend-domain>/healthz` should return `{"status":"ok"}`
- Run one sample through `/detect-accent` and ensure non-random output

## 6. Train models if needed
If `svm_model.pkl` and `label_encoder.pkl` do not exist, run training locally with your dataset and copy artifacts into `accsonify-backend/` before deploy.
