# CineTruth AI

CineTruth AI is a Streamlit-based media forensic screening application. It analyzes uploaded images, videos, and public media URLs for manipulation indicators, creates a weighted risk summary, stores media in Cloudflare R2, records optional telemetry in ClickHouse, and generates PDF screening reports.

> **Important:** CineTruth AI produces automated screening indicators. It does not prove that media is authentic, manipulated, stolen, or associated with a person. Human review and independent evidence are required for consequential decisions.

## Features

- Image and video uploads, including batch processing.
- Direct media URLs and supported public video-page URLs through `yt-dlp`.
- Cloudflare R2 storage for uploaded and URL-resolved media.
- Representative video frame extraction with OpenCV.
- Multimodal Google Gemini analysis for visual, audio/AV, and context signals.
- Efficient one-request Gemini pipeline and legacy full pipeline.
- Weighted local risk synthesis without an additional model request.
- SerpAPI Google Lens visual-search candidates.
- ImgBB temporary image hosting for reverse search uploads.
- ClickHouse telemetry, scan, takedown, and system-log persistence when configured.
- Downloadable PDF reports containing findings, scores, metadata, signals, and SHA-256 media fingerprints.

## Requirements

- Windows, macOS, or Linux.
- Python 3.10+ recommended.
- A Google Gemini API key for AI analysis.
- Cloudflare R2 credentials for the current media-upload workflow.
- Optional SerpAPI and ImgBB credentials for reverse image search.
- Optional ClickHouse credentials for persistence and telemetry.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency file includes Streamlit, Plotly, Google GenAI, OpenCV, MediaPipe, librosa, Pillow, ClickHouse Connect, boto3, ReportLab, yt-dlp, and imageio-ffmpeg.

## Configuration

Create a `.env` file in the project root. Never commit this file or expose its values in screenshots, logs, or reports.

### Required for AI analysis

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash
GEMINI_PIPELINE_MODE=efficient
```

`GEMINI_PIPELINE_MODE` accepts:

- `efficient`: one multimodal Gemini request per media item. This is the default and recommended mode.
- `full`: separate visual, audio/AV, and context agent calls, followed by local synthesis.

### Required for the current media workflow

```dotenv
R2_ACCOUNT_ID=your_cloudflare_account_id
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_bucket_name
```

Optional R2 settings:

```dotenv
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_PUBLIC_BASE_URL=https://media.example.com
R2_PRESIGNED_EXPIRY=3600
```

If `R2_PUBLIC_BASE_URL` is empty, private objects are served using presigned URLs. Media is limited to 100 MB and these extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.mov`, `.webm`, `.avi`, `.mkv`, and `.m4v`.

### Optional reverse image search

```dotenv
SERP_API_KEY=your_serpapi_key
IMGBB_API_KEY=your_imgbb_key
```

The application uploads a local reference image to ImgBB, then sends the resulting public URL to SerpAPI using the `google_lens` engine. A URL reference image can use SerpAPI directly without ImgBB.

### Optional ClickHouse

```dotenv
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_DATABASE=default
```

ClickHouse is disabled when `CLICKHOUSE_HOST` or `CLICKHOUSE_PASSWORD` is empty. The client connects over TLS and initializes project tables when credentials are available.

### General settings

```dotenv
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Run the application

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Open the local URL printed by Streamlit. The application has two tabs:

1. **Deepfake Media Detection**: upload one or more files or enter media URLs, resolve them into R2, run the forensic pipeline, review risk and consistency indicators, and download PDF reports.
2. **Identity Shield & Auto-Web Takedown**: submit a reference image, search for visual candidates through Google Lens/SerpAPI, and generate a factual human-review notice draft.

## Command-line pipeline

The pipeline can also process a local file without Streamlit:

```powershell
.venv\Scripts\python.exe main.py path\to\image.jpg
.venv\Scripts\python.exe main.py path\to\video.mp4
```

The command prints a JSON result to stdout. A valid local path and a supported extension are required.

## Processing pipeline

```text
Input file or URL
	|
	v
Cloudflare R2 upload / temporary URL resolution
	|
	v
MetadataAgent.extract_metadata()
	|
	+--> FrameAgent.extract_frames() for video samples
	|
	+--> FaceConsistencyAgent.analyze_faces()
	+--> AudioManipulationAgent.analyze_audio()
	+--> ContextVerificationAgent.verify_context()
	|
	v
GeminiMasterSynthesizer.synthesize_verdict()
	|
	v
PDF report + optional ClickHouse telemetry
```

### Efficient mode

`GeminiMasterSynthesizer.analyze_media_bundle()` sends one request containing the media and metadata. Gemini returns visual, audio/AV, context, and executive-summary fields in one JSON response. Local code then calculates the final weighted risk.

### Full mode

`Pipeline.execute()` calls the visual, audio, and context agents independently. `GeminiMasterSynthesizer.synthesize_verdict()` combines completed results locally and does not make another Gemini request.

## Models and scoring

The default model is read from `GEMINI_MODEL`, currently defaulting to `gemini-3.6-flash` in `config.py`. The model is used for:

- Visual forensic observations: face geometry, boundaries, lighting, reflections, texture, warping, duplication, and temporal artifacts.
- Audio/AV observations: voice-synthesis indicators, cadence, discontinuities, spectral changes, and lip-sync consistency.
- Context verification: internal consistency between media observations and extracted metadata.
- Optional takedown notice drafting.

All agent anomaly/risk values are normalized to `0.0` through `1.0`. The final manipulation-indicator risk is calculated locally:

```text
visual anomaly score: 45%
audio/AV anomaly score: 35%
context risk score: 20%
```

The final percentage is a weighted average of completed modules, scaled to `0`-`100`. If no module completes, the risk is `None`. A component match/consistency percentage is displayed as `100 - anomaly percentage`; it is not biometric identity matching.

Images skip audio analysis. Gemini quota failures return `QUOTA_EXCEEDED` with no risk percentage instead of inventing a score.

## Python API and methods

### `config.Config`

Environment-backed configuration class.

- `Config.clickhouse_configured() -> bool`: returns whether ClickHouse host and password are present.
- `Config.r2_configured() -> bool`: returns whether an R2 endpoint/account, access key, secret, and bucket are present.

Important attributes include `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_PIPELINE_MODE`, `SERP_API_KEY`, `IMGBB_API_KEY`, all `R2_*` settings, all `CLICKHOUSE_*` settings, and `TEMP_DIR`.

### `main.Pipeline`

- `Pipeline()`: creates frame, metadata, visual, audio, context, and master agents.
- `Pipeline.execute(media_path, source_label=None, session_id=None) -> dict`: runs one local image/video through the complete pipeline.

The result contains:

```python
{
    "session_id": str,
    "source": str,
    "media_path": str,
    "media_type": "image" | "video",
    "metadata": dict,
    "frames_sampled": int,
    "frame_paths": list[str],
    "agents": {
	"face_agent": dict,
	"audio_agent": dict,
	"context_agent": dict,
    },
    "final_verdict": dict,
    "pipeline_mode": "efficient" | "full",
}
```

### Forensic agents

`agents.frame_agent.FrameAgent`

- `FrameAgent(seconds_interval=2, max_frames=8)`: configures sampling.
- `extract_frames(video_path, output_dir="temp_frames") -> list[str]`: extracts representative JPEG frames with OpenCV.

`agents.metadata_agent.MetadataAgent`

- `extract_metadata(media_path) -> dict`: dispatches image or video metadata extraction.
- Image metadata includes EXIF fields, dimensions, format, and mode.
- Video metadata includes dimensions, FPS, frame count, duration, and file size.

`agents.face_agent.FaceConsistencyAgent`

- `analyze_faces(session_id, input_source="", frame_paths=None, source_label=None) -> dict`: analyzes image parts, sampled frames, or a video file through Gemini and logs the result.

`agents.audio_agent.AudioManipulationAgent`

- `analyze_audio(session_id, input_source="", source_label=None) -> dict`: analyzes video audio/AV consistency through Gemini; static images return `SKIPPED`.

`agents.context_agent.ContextVerificationAgent`

- `verify_context(session_id, evidence_summary) -> dict`: reviews existing evidence and returns a context risk score.

`agents.master_agent.GeminiMasterSynthesizer`

- `analyze_media_bundle(media_path, source_label, metadata, media_type) -> dict`: efficient one-request multimodal analysis.
- `synthesize_verdict(agent_outputs, model_summary="") -> dict`: local weighted risk synthesis.

### Cloudflare R2 API

`agents.media_storage.media_storage` is the shared `MediaStorage` instance.

- `MediaStorage.upload_bytes(data, filename, content_type=None) -> dict`
- `MediaStorage.upload_streamlit_file(uploaded_file) -> dict`
- `MediaStorage.upload_local_file(local_path, filename=None, content_type=None) -> dict`
- `MediaStorage.ingest_url(url) -> dict`: downloads a direct media URL or resolves a supported webpage with yt-dlp, then uploads it.
- `MediaStorage.get_access_url(key, expires_in=None) -> str`: returns a public or presigned URL.
- `MediaStorage.local_copy(key)`: context manager that downloads an R2 object to a temporary local file and deletes it afterward.
- `MediaStorage.delete(key)`: deletes an R2 object.
- `MediaStorage.check_connection() -> bool`: verifies bucket access.

Upload records contain `storage`, `bucket`, `key`, `filename`, `content_type`, `size`, and `sha256`.

### Reverse search and notices

`agents.takedown_agent.takedown_agent` is the shared `TakedownAgent` instance.

- `search_unauthorized_matches(input_data="User Identity") -> list`: accepts a public image URL, Streamlit uploaded file, or bytes; returns up to eight visual-search candidates.
- `generate_notice(target_url, similarity_score=None, notice_type="Content Review Request") -> dict`: creates an AI-generated or fallback factual notice draft and optionally logs it.

Candidate records include `target_url`, `matched_image_url`, `platform`, `title`, `similarity_score`, `thumbnail`, and `status`. SerpAPI/Google Lens does not provide a verified biometric match; `similarity_score` is intentionally `None`.

### PDF reports

`utils.report_generator.generate_report(data, output_filename="CineTruth_Forensic_Report.pdf") -> bool` creates a PDF from a `Pipeline.execute()` result. Reports include the executive assessment, component consistency, preview frame, agent findings, signals, technical metadata, SHA-256 fingerprint, pipeline status, and reader guidance.

### ClickHouse API

`database.clickhouse_db.db_manager` is the shared `ClickHouseManager` instance.

- `ClickHouseManager.save_scan(scan_id, input_type, source_path, media_type, verdict, confidence_score) -> bool`
- `ClickHouseManager.find_previous_scan(source_path) -> dict | None`
- `ClickHouseManager.log_agent_execution(session_id, agent_name, anomaly_score, status, details) -> bool`
- `ClickHouseManager.save_takedown_request(request_id, scan_id, platform, target_url, status="PENDING") -> bool`
- `ClickHouseManager.log_system_event(event_type, execution_time_ms=0, status="SUCCESS") -> bool`

All persistence methods return `False` when ClickHouse is unavailable rather than stopping local analysis.

## External API integrations

| Service | Purpose | Endpoint / SDK | Required? |
| --- | --- | --- | --- |
| Google Gemini | Multimodal forensic analysis and notice drafting | `google-genai`, configured model | Yes for AI scoring |
| Cloudflare R2 | Permanent media storage and access URLs | S3-compatible API through `boto3` | Yes for current UI upload flow |
| SerpAPI | Google Lens visual-search candidates | `https://serpapi.com/search`, `engine=google_lens` | Optional |
| ImgBB | Temporary public hosting of local reference images | `https://api.imgbb.com/1/upload` | Optional, needed for local-image reverse search |
| ClickHouse Cloud/self-hosted | Scan and agent telemetry | `clickhouse-connect` over TLS | Optional |
| yt-dlp | Public video-page resolution | Python package | Optional for webpage URLs |

## Project structure

```text
app.py                         Streamlit UI and workflow orchestration
main.py                        Pipeline class and CLI entry point
config.py                      .env-backed configuration
agents/
  master_agent.py              Gemini bundled analysis and local verdict synthesis
  face_agent.py                Visual forensic analysis
  audio_agent.py               Audio and AV analysis
  context_agent.py             Context consistency analysis
  frame_agent.py               OpenCV frame sampling
  metadata_agent.py            Image/video metadata extraction
  media_storage.py             Cloudflare R2 and URL ingestion
  takedown_agent.py            SerpAPI search and notice generation
  identity_agent.py            Placeholder identity API
database/
  clickhouse_db.py             Optional persistence and telemetry
utils/
  report_generator.py          PDF report generation
  video_processor.py            Minimal legacy video-processing helper
assets/style.css               Streamlit styling
requirements.txt               Python dependencies
```

## Placeholder identity module

`agents.identity_agent.IdentityAgent` is not used by the current Streamlit workflow and is not a production biometric system:

- `verify_identity_match(source_image, target_image)` currently returns a placeholder match and confidence.
- `extract_face_embeddings(image_bytes)` currently returns a fixed example vector.

Do not use these methods for identity decisions until they are replaced with a validated face-recognition implementation, consent controls, threshold calibration, and privacy safeguards.

## Troubleshooting

### Gemini errors or no risk score

Check `GEMINI_API_KEY` and `GEMINI_MODEL`. A daily free-tier quota error is reported as `QUOTA_EXCEEDED`; wait for reset, use an available model/project, or switch credentials.

### R2 upload errors

Check all four required R2 variables, bucket permissions, and the endpoint. The UI requires R2 before it accepts media for the current batch workflow.

### Reverse search returns no candidates

Check both `SERP_API_KEY` and, for local uploads, `IMGBB_API_KEY`. Public image URLs can bypass ImgBB. Search results are candidates only and may be empty or unrelated.

### Video URL cannot be resolved

Confirm the URL is public and supported by yt-dlp. Separate video/audio streams may require the bundled `imageio-ffmpeg` executable; private, login-protected, or blocked pages may fail.

### ClickHouse is unavailable

ClickHouse is optional. Local forensic processing continues without persistence, and telemetry methods return `False` when no client is connected.

## Privacy and operational guidance

- Keep API credentials in `.env` or a secret manager, never in source control.
- Treat uploaded media and reference images as sensitive data.
- Review R2 bucket visibility, presigned URL expiry, retention, and deletion policies.
- Preserve original evidence separately from derived frames and reports.
- Do not treat model output, visual-search candidates, or the placeholder identity module as proof of identity, authorship, infringement, or manipulation.
