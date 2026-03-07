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

## Netlify deploy

This repository is configured with root `netlify.toml`:

- Base directory: `accsonify-frontend`
- Build command: `npm run build`
- Next.js plugin: `@netlify/plugin-nextjs`

In Netlify site settings, add environment variable:

- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-domain>`

Important:

- The ML backend (Whisper + Torch + FastAPI) should run on Docker host (Render/Railway/VM).
- Netlify should host the frontend only.
