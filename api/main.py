import os
import secrets
import time
from collections import deque
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from src.url_validator import validate_url
from src.url_detector import analyze_url
from src.ml_detector import predict_url
from src.email_detector import analyze_email


app = FastAPI(
    title="AI Cybersecurity Guardian API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class URLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("CYBER_GUARDIAN_API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# Development-only, in-memory rate limiter.
# Replace with a shared backend before multi-worker deployment.
RATE_LIMIT = 10
RATE_WINDOW_SECONDS = 60
_request_times = {}
_rate_lock = Lock()


def enforce_rate_limit(
    x_api_key: str | None = Header(default=None),
):
    # Authentication must succeed before this dependency runs.
    expected = os.environ.get("CYBER_GUARDIAN_API_KEY")

    if not expected or not x_api_key or not secrets.compare_digest(
        x_api_key, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    now = time.monotonic()

    # Hash the credential so raw API keys are not stored in this dictionary.
    import hashlib
    key_id = hashlib.sha256(x_api_key.encode()).hexdigest()

    with _rate_lock:
        timestamps = _request_times.setdefault(key_id, deque())

        while timestamps and now - timestamps[0] >= RATE_WINDOW_SECONDS:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT:
            remaining = max(
                1,
                int(RATE_WINDOW_SECONDS - (now - timestamps[0])) + 1,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(remaining)},
            )

        timestamps.append(now)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/api/v1/analyze/url",
    dependencies=[
        Depends(require_api_key),
        Depends(enforce_rate_limit),
    ],
)
def analyze_url_endpoint(request: URLRequest):
    validation = validate_url(request.url)

    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=validation["message"],
        )

    normalized_url = validation["url"]
    findings = analyze_url(normalized_url)
    try:
        ml_result = predict_url(normalized_url)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("ML inference failed")
        raise HTTPException(
            status_code=503,
            detail="Analysis service temporarily unavailable.",
        ) from None

    return {
        "status": "analyzed",
        "normalized": validation["normalized"],
        "findings": findings,
        "ml_analysis": {
            "classification": (
                "flagged"
                if ml_result["phishing_score"] >= ml_result["threshold"]
                else "not_flagged"
            ),
            "phishing_class_score": ml_result["phishing_score"],
            "threshold": ml_result["threshold"],
            "model": ml_result["model"],
        },
        "disclaimer": (
            "ML score is not a calibrated real-world phishing probability. "
            "A URL that is not flagged is not necessarily safe."
        ),
    }


class EmailRequest(BaseModel):
    email_text: str = Field(min_length=1, max_length=50000)


@app.post(
    "/api/v1/analyze/email",
    dependencies=[
        Depends(require_api_key),
        Depends(enforce_rate_limit),
    ],
)
def analyze_email_endpoint(request: EmailRequest):
    if not request.email_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Email text cannot be empty.",
        )

    result = analyze_email(request.email_text)

    return {
        "status": "analyzed",
        **result,
        "disclaimer": (
            "Email risk score is heuristic, not a calibrated "
            "phishing probability. A low score does not "
            "guarantee that an email is safe."
        ),
    }
