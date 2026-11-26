import os
import secrets
import time
from collections import deque
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

from datetime import date, datetime, timezone

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.url_validator import validate_url
from src.url_detector import analyze_url
from src.ml_detector import predict_url
from src.email_detector import analyze_email, extract_email_urls


app = FastAPI(
    title="AI Cybersecurity Guardian API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# Public demo mode (for a hosted copy). Visitors do not need an API key, so the
# limits below protect the server: a per-visitor rate limit and a shared daily cap.
# Off by default, so local Docker use keeps full API-key authentication.
def demo_mode() -> bool:
    return os.environ.get("DEMO_MODE", "").lower() in {"1", "true", "yes"}


DEMO_DAILY_LIMIT = int(os.environ.get("DEMO_DAILY_REQUESTS", "500"))
_demo_day = datetime.now(timezone.utc).date()
_demo_used = 0
_demo_lock = Lock()


def spend_demo_budget() -> bool:
    """Count one request against today's shared budget (resets at midnight UTC)."""
    global _demo_day, _demo_used
    with _demo_lock:
        today = datetime.now(timezone.utc).date()
        if today != _demo_day:
            _demo_day, _demo_used = today, 0
        if _demo_used >= DEMO_DAILY_LIMIT:
            return False
        _demo_used += 1
        return True


def visitor_id(request: Request) -> str:
    """Behind Render's proxy the real address is the first X-Forwarded-For entry."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


class URLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


def require_api_key(x_api_key: str | None = Header(default=None)):
    if demo_mode():
        return

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
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    import hashlib

    if demo_mode():
        # No key in demo mode: limit each visitor by address, then check the shared budget.
        identity = "demo:" + visitor_id(request)
        if not spend_demo_budget():
            raise HTTPException(
                status_code=429,
                detail="The demo has reached its limit for today. Please try again "
                "tomorrow, or run it yourself from the GitHub repository.",
            )
    else:
        # Authentication must succeed before this dependency runs.
        expected = os.environ.get("CYBER_GUARDIAN_API_KEY")

        if not expected or not x_api_key or not secrets.compare_digest(
            x_api_key, expected
        ):
            raise HTTPException(
                status_code=401,
                detail="Authentication required.",
            )
        identity = x_api_key

    now = time.monotonic()

    # Hash the identity so raw API keys and addresses are not stored in this dictionary.
    key_id = hashlib.sha256(identity.encode()).hexdigest()

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


@app.get("/api/v1/config")
def public_config():
    """Lets the page know whether to show the demo banner. Contains nothing secret."""
    return {"demo": demo_mode(), "daily_limit": DEMO_DAILY_LIMIT}


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

    # Analyze URLs embedded in the email independently.
    embedded_url_analysis = []

    for embedded_url in extract_email_urls(request.email_text):
        validation = validate_url(embedded_url)

        if not validation["valid"]:
            embedded_url_analysis.append({
                "url": embedded_url,
                "status": "invalid",
                "findings": [validation["message"]],
                "ml_analysis": None,
            })
            continue

        normalized_url = validation["url"]
        url_findings = analyze_url(normalized_url)

        try:
            ml_result = predict_url(normalized_url)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Embedded URL ML inference failed"
            )
            embedded_url_analysis.append({
                "url": embedded_url,
                "status": "unavailable",
                "findings": url_findings,
                "ml_analysis": None,
            })
            continue

        embedded_url_analysis.append({
            "url": embedded_url,
            "status": "analyzed",
            "findings": url_findings,
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
        })

    return {
        "status": "analyzed",
        **result,
        "embedded_url_analysis": embedded_url_analysis,
        "disclaimer": (
            "Email risk score is heuristic, not a calibrated "
            "phishing probability. A low score does not "
            "guarantee that an email is safe."
        ),
    }


# Serve the built frontend last, so it never hides an API route (single-container deploy).
_static_dir = os.environ.get("STATIC_DIR", "")
if _static_dir and os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="web")
