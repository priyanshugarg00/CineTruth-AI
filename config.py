import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    # Primary model. If Streamlit secrets already set GEMINI_MODEL, that value remains primary.
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    # Comma-separated stable fallbacks. Duplicates are removed at runtime.
    GEMINI_FALLBACK_MODELS = tuple(
        model.strip()
        for model in os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite",
        ).split(",")
        if model.strip()
    )
    # Number of generate_content attempts PER model, including the first request.
    GEMINI_RETRY_ATTEMPTS = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "3"))
    GEMINI_FILE_RETRY_ATTEMPTS = int(os.getenv("GEMINI_FILE_RETRY_ATTEMPTS", "3"))
    GEMINI_RETRY_INITIAL_DELAY = float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", "1.5"))
    GEMINI_RETRY_MAX_DELAY = float(os.getenv("GEMINI_RETRY_MAX_DELAY", "8"))
    # efficient = one logical Gemini analysis per scan; retries/fallback attempts may make multiple API calls.
    GEMINI_PIPELINE_MODE = os.getenv("GEMINI_PIPELINE_MODE", "efficient").strip().lower()

    # Reverse image search
    SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "").strip()

    # Cloudflare R2 permanent media storage
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()
    # Optional override, useful for jurisdiction-specific R2 endpoints.
    R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
    # Optional public/custom-domain base URL. Private buckets work with presigned URLs.
    R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "").strip()
    R2_PRESIGNED_EXPIRY = int(os.getenv("R2_PRESIGNED_EXPIRY", "3600"))

    # ClickHouse is OPTIONAL until credentials are configured.
    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "").strip()
    CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8443"))
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default").strip()
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "").strip()
    CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default").strip()

    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Backwards-compatible processing directory used only for short-lived frames/reports.
    # Uploaded image/video persistence is handled by Cloudflare R2.
    TEMP_DIR = os.path.join(tempfile.gettempdir(), "cinetruth_ai_processing")

    @classmethod
    def clickhouse_configured(cls) -> bool:
        return bool(cls.CLICKHOUSE_HOST and cls.CLICKHOUSE_PASSWORD)

    @classmethod
    def r2_configured(cls) -> bool:
        endpoint_ready = bool(cls.R2_ENDPOINT_URL or cls.R2_ACCOUNT_ID)
        return bool(
            endpoint_ready
            and cls.R2_ACCESS_KEY_ID
            and cls.R2_SECRET_ACCESS_KEY
            and cls.R2_BUCKET_NAME
        )


os.makedirs(Config.TEMP_DIR, exist_ok=True)
