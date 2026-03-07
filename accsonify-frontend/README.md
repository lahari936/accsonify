# Accsonify Frontend

Next.js frontend for accent detection and conversion.

## Local run

1. Install dependencies:

```bash
npm install
```

2. Set environment variable in `.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

3. Run development server:

```bash
npm run dev
```

Open `http://localhost:3000`.

## Deployment env vars

Netlify:

```bash
NEXT_PUBLIC_API_BASE_URL=https://<your-backend-domain>
```

Backend:

```bash
FRONTEND_ORIGIN=https://<your-netlify-domain>
ALLOW_MOCK_MODE=true
```

For real inference:

1. Place `svm_model.pkl` and `label_encoder.pkl` in `accsonify-backend/`.
2. Set:

```bash
ALLOW_MOCK_MODE=false
```
