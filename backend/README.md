# Backend API Authentication

All endpoints under `/api` require an API key in the `X-API-Key` header.

## Required environment variable

Set `API_KEY` before starting the backend:

```bash
export API_KEY="your-secret-key"
```

## Required request header

Include this header on protected requests:

```http
X-API-Key: your-secret-key
```

If the header is missing or invalid, the API returns `401 Unauthorized`.

## Public allowlist

`GET /api/health` is intentionally public for monitoring/liveness checks and does not require `X-API-Key`.
