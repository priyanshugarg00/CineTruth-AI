import json
import os
import time

from google import genai

from config import Config
from utils.gemini_client import (
    GeminiRequestFailure,
    generate_content_with_fallback,
    run_transient_operation,
)
from database.clickhouse_db import db_manager


class AudioManipulationAgent:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def __init__(self):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY) if Config.GEMINI_API_KEY else None

    @staticmethod
    def _parse_json(text: str) -> dict:
        clean = (text or "").replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
        return json.loads(clean)

    def analyze_audio(self, session_id: str, input_source: str = "", source_label: str | None = None) -> dict:
        source_label = source_label or input_source
        ext = os.path.splitext(str(input_source))[1].lower()

        if ext not in self.VIDEO_EXTENSIONS:
            result = {
                "agent": "Audio Manipulation Agent",
                "status": "SKIPPED",
                "anomaly_score": 0.0,
                "details": "Static image source: no audio track to analyze.",
                "signals": [],
                "gemini_requests_used": 0,
                "gemini_models_tried": [],
            }
        elif not self.client:
            result = {
                "agent": "Audio Manipulation Agent",
                "status": "ERROR",
                "anomaly_score": 0.0,
                "details": "GEMINI_API_KEY is not configured.",
                "signals": [],
                "gemini_requests_used": 0,
                "gemini_models_tried": [],
            }
        elif not os.path.exists(input_source):
            result = {
                "agent": "Audio Manipulation Agent",
                "status": "ERROR",
                "anomaly_score": 0.0,
                "details": f"Video path not found: {input_source}",
                "signals": [],
                "gemini_requests_used": 0,
                "gemini_models_tried": [],
            }
        else:
            try:
                uploaded = run_transient_operation(
                    lambda: self.client.files.upload(file=input_source),
                    operation_name="Gemini video upload",
                )
                for _ in range(30):
                    state = getattr(uploaded, "state", None)
                    state_name = getattr(state, "name", str(state or ""))
                    if state_name != "PROCESSING":
                        break
                    time.sleep(1)
                    uploaded = run_transient_operation(
                        lambda: self.client.files.get(name=uploaded.name),
                        operation_name="Gemini file status check",
                    )

                prompt = f"""
You are CineTruth AI's audio/AV forensic agent. Analyze the audio and audio-video alignment in this video from: {source_label}.
Look for suspicious voice synthesis artifacts, unnatural cadence, abrupt spectral/voice changes, dubbing inconsistencies,
or visible lip-sync mismatch. Do not claim certainty.
Return ONLY valid JSON:
{{
  "anomaly_score": 0.0,
  "details": "Concise evidence-based audio/AV finding",
  "signals": ["signal 1", "signal 2"]
}}
`anomaly_score` must be between 0 and 1 and represents manipulation-indicator risk only.
"""
                response, gemini_meta = generate_content_with_fallback(
                    self.client,
                    contents=[uploaded, prompt],
                    primary_model=Config.GEMINI_MODEL,
                )
                data = self._parse_json(response.text)
                score = max(0.0, min(1.0, float(data.get("anomaly_score", 0.0))))
                signals = data.get("signals", [])
                if not isinstance(signals, list):
                    signals = [str(signals)]
                result = {
                    "agent": "Audio Manipulation Agent",
                    "status": "COMPLETED",
                    "anomaly_score": score,
                    "details": str(data.get("details", "Audio analysis completed.")),
                    "signals": [str(x) for x in signals[:8]],
                    "gemini_requests_used": gemini_meta.requests_used,
                    "gemini_model_used": gemini_meta.model_used,
                    "gemini_models_tried": gemini_meta.models_tried,
                }
            except GeminiRequestFailure as exc:
                result = {
                    "agent": "Audio Manipulation Agent",
                    "status": exc.kind,
                    "anomaly_score": 0.0,
                    "details": exc.public_message,
                    "signals": [],
                    "gemini_requests_used": exc.requests_used,
                    "gemini_models_tried": exc.models_tried,
                    "technical_error": exc.technical_error,
                }
            except Exception as exc:
                result = {
                    "agent": "Audio Manipulation Agent",
                    "status": "ERROR",
                    "anomaly_score": 0.0,
                    "details": f"Audio/AV analysis failed: {exc}",
                    "signals": [],
                    "gemini_requests_used": 0,
                    "gemini_models_tried": [],
                }

        db_manager.log_agent_execution(
            session_id=session_id,
            agent_name=result["agent"],
            anomaly_score=result["anomaly_score"],
            status=result["status"],
            details=result["details"],
        )
        return result
