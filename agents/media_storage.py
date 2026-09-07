from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import boto3
import requests
from botocore.client import Config as BotoConfig

from config import Config


class MediaStorage:
    """Cloudflare R2 media storage for CineTruth AI.

    Permanent image/video uploads live in R2. A short-lived local file is created
    only when OpenCV/Gemini needs a filesystem path for processing. That file is
    deleted automatically after the analysis block finishes.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    MAX_MEDIA_BYTES = 100 * 1024 * 1024

    CONTENT_TYPE_EXTENSIONS = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
    }

    def __init__(self):
        self.bucket = Config.R2_BUCKET_NAME
        self.public_base_url = Config.R2_PUBLIC_BASE_URL.rstrip("/")
        self._client = None

        if self.configured:
            endpoint = Config.R2_ENDPOINT_URL or (
                f"https://{Config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
            )
            self._client = boto3.client(
                service_name="s3",
                endpoint_url=endpoint,
                aws_access_key_id=Config.R2_ACCESS_KEY_ID,
                aws_secret_access_key=Config.R2_SECRET_ACCESS_KEY,
                region_name="auto",
                config=BotoConfig(signature_version="s3v4"),
            )

    @property
    def configured(self) -> bool:
        endpoint_ready = bool(Config.R2_ENDPOINT_URL or Config.R2_ACCOUNT_ID)
        return bool(
            endpoint_ready
            and Config.R2_ACCESS_KEY_ID
            and Config.R2_SECRET_ACCESS_KEY
            and Config.R2_BUCKET_NAME
        )

    def _require_config(self):
        if not self.configured or self._client is None:
            raise RuntimeError(
                "Cloudflare R2 is not configured. Set R2_ACCOUNT_ID, "
                "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME in .env."
            )

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = os.path.basename(filename or "media")
        stem, ext = os.path.splitext(name)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "media"
        ext = re.sub(r"[^A-Za-z0-9.]", "", ext.lower())
        return f"{stem[:80]}{ext}"

    def _object_key(self, filename: str, prefix: str = "forensic-media") -> str:
        now = datetime.now(timezone.utc)
        safe_name = self._safe_filename(filename)
        return (
            f"{prefix}/{now:%Y/%m/%d}/"
            f"{uuid.uuid4().hex[:12]}_{safe_name}"
        )

    def _validate_extension(self, filename: str):
        ext = Path(filename).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported media extension: {ext or 'unknown'}")

    def _record(self, key: str, filename: str, content_type: str, size: int, sha256: str) -> dict:
        return {
            "storage": "cloudflare_r2",
            "bucket": self.bucket,
            "key": key,
            "filename": filename,
            "content_type": content_type,
            "size": int(size),
            "sha256": sha256,
        }

    def upload_bytes(self, data: bytes, filename: str, content_type: str | None = None) -> dict:
        self._require_config()
        self._validate_extension(filename)
        if not data:
            raise ValueError("Cannot upload an empty media file.")
        if len(data) > self.MAX_MEDIA_BYTES:
            raise ValueError("Media is larger than the 100 MB test limit.")

        safe_name = self._safe_filename(filename)
        key = self._object_key(safe_name)
        content_type = content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        digest = hashlib.sha256(data).hexdigest()

        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={
                "original-filename": safe_name,
                "sha256": digest,
            },
        )
        return self._record(key, safe_name, content_type, len(data), digest)

    def upload_streamlit_file(self, uploaded_file) -> dict:
        """Upload a Streamlit UploadedFile directly to R2."""
        data = uploaded_file.getvalue()
        return self.upload_bytes(
            data=data,
            filename=uploaded_file.name,
            content_type=getattr(uploaded_file, "type", None),
        )

    def upload_local_file(self, local_path: str, filename: str | None = None, content_type: str | None = None) -> dict:
        """Upload an already-resolved temporary media file to R2."""
        self._require_config()
        if not local_path or not os.path.isfile(local_path):
            raise FileNotFoundError(f"Media file not found: {local_path}")

        size = os.path.getsize(local_path)
        if size > self.MAX_MEDIA_BYTES:
            raise ValueError("Media is larger than the 100 MB test limit.")

        filename = filename or os.path.basename(local_path)
        self._validate_extension(filename)
        safe_name = self._safe_filename(filename)
        key = self._object_key(safe_name)
        content_type = content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

        digest = hashlib.sha256()
        with open(local_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)

        self._client.upload_file(
            local_path,
            self.bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": {
                    "original-filename": safe_name,
                    "sha256": digest.hexdigest(),
                },
            },
        )
        return self._record(key, safe_name, content_type, size, digest.hexdigest())

    def get_access_url(self, key: str, expires_in: int | None = None) -> str:
        """Return a public URL when configured, otherwise a temporary presigned URL."""
        self._require_config()
        if self.public_base_url:
            return f"{self.public_base_url}/{quote(key, safe='/')}"

        expiry = expires_in or Config.R2_PRESIGNED_EXPIRY
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=int(expiry),
        )

    @contextmanager
    def local_copy(self, key: str):
        """Temporarily download an R2 object for OpenCV/Gemini processing."""
        self._require_config()
        suffix = Path(key).suffix.lower()
        fd, path = tempfile.mkstemp(prefix="cinetruth_media_", suffix=suffix)
        os.close(fd)
        try:
            self._client.download_file(self.bucket, key, path)
            yield path
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def delete(self, key: str):
        self._require_config()
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def check_connection(self) -> bool:
        """Verify that the configured credentials can access the target bucket."""
        self._require_config()
        self._client.list_objects_v2(Bucket=self.bucket, MaxKeys=1)
        return True

    def _download_with_ytdlp(self, url: str, work_dir: str) -> str:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError(
                "This URL is a webpage, not a direct media file. Install yt-dlp with: pip install yt-dlp"
            ) from exc

        token = uuid.uuid4().hex[:10]
        output_template = os.path.join(work_dir, f"url_{token}.%(ext)s")
        ydl_opts = {
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "max_filesize": self.MAX_MEDIA_BYTES,
            "socket_timeout": 40,
            "retries": 3,
            "fragment_retries": 3,
            "merge_output_format": "mp4",
        }

        try:
            import imageio_ffmpeg

            ydl_opts["ffmpeg_location"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise RuntimeError("No downloadable media was found on this URL.")
        except Exception as exc:
            raise RuntimeError(
                "The URL opened as a webpage, but its video could not be downloaded. "
                "The site may require login/cookies, block automated downloads, or the post may be private. "
                f"Details: {exc}"
            ) from exc

        candidates = []
        for name in os.listdir(work_dir):
            if not name.startswith(f"url_{token}."):
                continue
            path = os.path.join(work_dir, name)
            if os.path.isfile(path) and Path(path).suffix.lower() in self.ALLOWED_EXTENSIONS:
                candidates.append(path)

        if not candidates:
            raise RuntimeError("The page was resolved, but no supported image/video file was produced.")

        target = max(candidates, key=os.path.getsize)
        if os.path.getsize(target) > self.MAX_MEDIA_BYTES:
            raise ValueError("Resolved media is larger than the 100 MB test limit.")
        return target

    def ingest_url(self, url: str) -> dict:
        """Resolve a direct media/video-page URL and persist the resulting media in R2."""
        self._require_config()
        parsed = urlparse((url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Please enter a valid http:// or https:// media URL.")

        with tempfile.TemporaryDirectory(prefix="cinetruth_url_") as work_dir:
            response = requests.get(
                url,
                timeout=40,
                stream=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0 Safari/537.36"
                    )
                },
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            resolved_path = urlparse(response.url).path
            ext = Path(resolved_path).suffix.lower()

            if content_type in {"text/html", "application/xhtml+xml"} or content_type.startswith("text/html"):
                response.close()
                local_path = self._download_with_ytdlp(url, work_dir)
                filename = os.path.basename(local_path)
                return self.upload_local_file(local_path, filename=filename)

            if ext not in self.ALLOWED_EXTENSIONS:
                ext = self.CONTENT_TYPE_EXTENSIONS.get(content_type, "")
            if ext not in self.ALLOWED_EXTENSIONS:
                response.close()
                raise ValueError(
                    f"Unsupported media type from URL: {content_type or 'unknown'}. "
                    "Use a direct image/video URL or a supported public video-page URL."
                )

            filename = f"url_{uuid.uuid4().hex[:10]}{ext}"
            local_path = os.path.join(work_dir, filename)
            total = 0
            try:
                with open(local_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.MAX_MEDIA_BYTES:
                            raise ValueError("Media URL is larger than the 100 MB test limit.")
                        fh.write(chunk)
            finally:
                response.close()

            return self.upload_local_file(
                local_path,
                filename=filename,
                content_type=content_type or None,
            )


media_storage = MediaStorage()
