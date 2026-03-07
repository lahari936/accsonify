# Netlify Deployment Guide

This project should be deployed as:
- Frontend on Netlify
- Backend (FastAPI + Whisper + Torch) on Docker host

## 1. Backend first (required)

Deploy backend from `accsonify-backend/` with Docker and ensure these files exist in backend folder:
- `svm_model.pkl`
- `label_encoder.pkl`

Check backend health:

```bash
curl https://<your-backend-domain>/healthz
```

## 2. Connect repo to Netlify

In Netlify, create site from this GitHub repository.

Repository config is already provided in root `netlify.toml`:
- Base: `accsonify-frontend`
- Command: `npm run build`
- Plugin: `@netlify/plugin-nextjs`

## 3. Add required env var in Netlify

Site settings -> Environment variables:

- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-domain>`

## 4. Deploy

Trigger deploy from Netlify UI.

After deploy, validate:
- Frontend loads
- `detect-accent`, `transcribe`, and `convert-accent` calls hit backend correctly

## 5. Notes

- Do not deploy heavy ML inference on Netlify functions for this project.
- Keep backend CORS `FRONTEND_ORIGIN` aligned with your Netlify domain.
