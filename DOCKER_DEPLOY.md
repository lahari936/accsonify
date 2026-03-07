# Accsonify Docker Deployment (Exact ML Pipeline)

This setup runs:
- FastAPI backend (`accsonify-backend`) with Whisper + SVM inference
- Next.js frontend (`accsonify-frontend`)

## 1. Prepare dataset and train models

Your training script now uses the exact pipeline you shared.

Important: `c:\Users\Lahari\Downloads\Unconfirmed 589443.crdownload` is an incomplete download format. Rename/extract only after download finishes and you have the real dataset folder.

Expected dataset structure:

- `<DATASET_BASE>/speakers_all.csv`
- `<DATASET_BASE>/recordings/recordings/*.mp3`

Train from repository root:

```bash
python accsonify-backend/train_improved_model.py --base-path "C:/path/to/speech-accent-archive"
```

This creates:
- `accsonify-backend/svm_model.pkl`
- `accsonify-backend/label_encoder.pkl`

## 2. Build and run website with Docker

From repository root (`Implementation`):

```bash
docker compose build
docker compose up -d
```

App URLs:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Health check: `http://localhost:8000/healthz`

## 3. Stop

```bash
docker compose down
```

## 4. Production notes

- Keep `ALLOW_MOCK_MODE=false`.
- Ensure trained model files are present in `accsonify-backend/` before image build.
- Set frontend build arg `NEXT_PUBLIC_API_BASE_URL` to your backend public URL.
