# AI Cybersecurity Guardian

AI Cybersecurity Guardian is a locally deployed phishing URL analysis application combining rule-based URL inspection with a machine-learning classifier.

The application provides a React dashboard backed by a FastAPI service and a trained Random Forest model.

> **Project status:** Working local prototype. Not hardened or validated for public production deployment.

## Architecture

```text
Browser
   |
   v
React frontend served by Nginx
   |
   | /api/ requests
   v
FastAPI backend
   |
   +-- URL validation and normalization
   +-- Rule-based URL analysis
   +-- Feature extraction
   +-- Random Forest inference
   |
   v
Analysis result returned to dashboard
```

Docker Compose runs two services on an internal bridge network:

- `frontend`: Nginx serving the compiled React application and proxying API requests.
- `backend`: FastAPI application with the trained ML model.

Only the frontend is published to the host, at `127.0.0.1:8080`.

## Technology Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite |
| Web server / proxy | Nginx |
| Backend | Python 3.13, FastAPI, Uvicorn |
| Machine learning | scikit-learn Random Forest |
| Data processing | Pandas, NumPy |
| Model serialization | joblib |
| Containerization | Docker, Docker Compose |

## Local Docker Setup

### Requirements

- Docker with Docker Compose support
- An available local port `8080`

### 1. Configure the environment

From the project root:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with a unique, randomly generated API key:

```dotenv
CYBER_GUARDIAN_API_KEY=your-unique-random-secret
```

Never commit the real `.env` file or share its API key.

### 2. Build and start

```bash
docker compose up -d --build
```

### 3. Open the dashboard

Visit:

http://127.0.0.1:8080

### 4. Stop the application

```bash
docker compose down
```

## API

### Health check

```http
GET /health
```

Successful response:

```json
{"status": "ok"}
```

### Analyze a URL

```http
POST /api/v1/analyze/url
Content-Type: application/json
X-API-Key: <configured-secret>
```

Request body:

```json
{
  "url": "https://example.com"
}
```

A successful response includes:

- Analysis status and normalization information
- Rule-based findings
- ML classification: `flagged` or `not_flagged`
- Phishing-class score
- Classification threshold
- Model name
- A warning about score interpretation

The backend requires an API key for the analysis endpoint. In the local Docker setup, Nginx injects the configured key when proxying browser requests.

The API currently uses an in-memory rate limit of 10 requests per 60 seconds per API key.

Interactive FastAPI documentation and the OpenAPI endpoint are disabled in this implementation.

## Email Phishing Analysis

The application analyzes email content, suspicious sender domains, brand impersonation, and embedded URLs.

### Email API

Endpoint: `POST /api/v1/analyze/email`

Request body: `email_text` (maximum 50,000 characters).

The endpoint requires an API key and returns email risk findings, sender analysis, and independent analysis of embedded URLs using URL rules and machine learning.

Email risk scores are heuristic, not calibrated phishing probabilities. SPF, DKIM, and DMARC authentication are not independently verified.

## Machine-Learning Model

The deployed model is stored at:

```text
models/diverse/random_forest_diverse.pkl
```

It is identified in the API as `Diverse Random Forest`.

The model analyzes 11 numerical URL features:

| Feature | Description |
| --- | --- |
| `url_length` | Total URL length |
| `hostname_length` | Hostname length |
| `dot_count` | Number of dots in the URL |
| `hyphen_count` | Number of hyphens |
| `digit_count` | Number of digits |
| `special_char_count` | Count of selected special characters |
| `https` | HTTPS scheme indicator |
| `has_ip` | IP-address hostname indicator |
| `subdomain_dots` | Number of dots in the hostname |
| `path_length` | URL path length |
| `query_length` | Query-string length |

Inference uses `predict_proba()` and a classification threshold of `0.50`.

A score greater than or equal to `0.50` is classified as `flagged`.

The model is loaded once per Python process using an in-memory cache.

## Testing

Backend API and inference tests are located under `tests/`.

Additional scripts in the repository cover model evaluation, feature ablation, calibration, dataset auditing, and other research experiments.

Research evaluation results are stored under `results/` and `models/`.

Performance figures should be interpreted in the context of the specific evaluation datasets and experimental protocols.

## Security and Research Limitations

- The ML classifier uses URL structure, not webpage content or live website behavior.
- A URL classified as `not_flagged` is not necessarily safe.
- The phishing-class score is not a calibrated real-world probability.
- HTTPS does not guarantee that a website is legitimate.
- Dataset-specific shortcuts and distribution shifts can affect generalization.
- The local Nginx proxy uses a shared API key for browser requests.
- The in-memory rate limiter is not a distributed, per-user abuse-control system.
- Public deployment would require additional authentication, traffic protection, operational monitoring, and security review.

## Repository Notes

- `.env` contains local secrets and must remain excluded from Git.
- `.env.example` is a safe configuration template.
- `data/` is excluded from Git because research datasets may be large or subject to separate distribution conditions.
- Serialized model artifacts are included for the current local deployment.

## Disclaimer

AI Cybersecurity Guardian is a research and educational prototype. Its output should be treated as a risk indicator, not a definitive determination that a URL is safe or malicious.
