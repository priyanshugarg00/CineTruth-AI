import json
import mimetypes
import os
import time

from google import genai
from google.genai import types

from config import Config
from database.clickhouse_db import db_manager


class FaceConsistencyAgent:
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

    def _analyze_image_parts(self, image_paths: list[str], source_label: str) -> dict:
        parts = []
        for path in image_paths[:8]:
            if not os.path.exists(path):
                continue
            mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as fh:
                parts.append(types.Part.from_bytes(data=fh.read(), mime_type=mime_type))

        if not parts:
            raise ValueError("No readable image/frame data was supplied.")

        prompt = f"""
You are CineTruth AI's visual forensic agent. Inspect the supplied image or sampled video frames from: {source_label}
Look for visible manipulation indicators such as inconsistent facial geometry, warped boundaries, lighting/reflection mismatch,
unnatural skin texture, temporal/frame inconsistencies, duplicated regions, or generation artifacts.
Do NOT claim certainty. Return ONLY valid JSON:
{{
  "anomaly_score": 0.0,
  "details": "Concise evidence-based visual finding",
  "signals": ["signal 1", "signal 2"]
}}
`anomaly_score` must be between 0 and 1 and should represent manipulation-indicator risk, not proof of a deepfake.
"""
        response = self.client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=[*parts, prompt],
        )
        data = self._parse_json(response.text)
        score = max(0.0, min(1.0, float(data.get("anomaly_score", 0.0))))
        signals = data.get("signals", [])
        if not isinstance(signals, list):
            signals = [str(signals)]
        return {
            "anomaly_score": score,
            "details": str(data.get("details", "Visual analysis completed.")),
            "signals": [str(x) for x in signals[:8]],
        }

    def _analyze_video_file(self, video_path: str, source_label: str) -> dict:
        uploaded = self.client.files.upload(file=video_path)

        # Video files can require server-side processing before Gemini can inspect them.
        for _ in range(30):
            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", str(state or ""))
            if state_name != "PROCESSING":
                break
            time.sleep(1)
            uploaded = self.client.files.get(name=uploaded.name)

        prompt = f"""
You are CineTruth AI's visual forensic agent. Analyze the visual stream in this video from: {source_label}.
Inspect face boundaries, temporal consistency, lighting, reflections, motion/warping, texture artifacts and obvious synthetic-media indicators.
Do NOT claim certainty. Return ONLY valid JSON:
{{
  "anomaly_score": 0.0,
  "details": "Concise evidence-based visual finding",
  "signals": ["signal 1", "signal 2"]
}}
`anomaly_score` must be between 0 and 1 and represents manipulation-indicator risk only.
"""
        response = self.client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=[uploaded, prompt],
        )
        data = self._parse_json(response.text)
        score = max(0.0, min(1.0, float(data.get("anomaly_score", 0.0))))
        signals = data.get("signals", [])
        if not isinstance(signals, list):
            signals = [str(signals)]
        return {
            "anomaly_score": score,
            "details": str(data.get("details", "Visual analysis completed.")),
            "signals": [str(x) for x in signals[:8]],
        }

    def analyze_faces(
        self,
        session_id: str,
        input_source: str = "",
        frame_paths: list[str] | None = None,
        source_label: str | None = None,
    ) -> dict:
        source_label = source_label or input_source

        if not self.client:
            result = {
                "agent": "Face Consistency Agent",
                "status": "ERROR",
                "anomaly_score": 0.0,
                "details": "GEMINI_API_KEY is not configured.",
                "signals": [],
            }
        else:
            try:
                if frame_paths:
                    analysis = self._analyze_image_parts(frame_paths, source_label)
                elif os.path.exists(input_source):
                    ext = os.path.splitext(input_source)[1].lower()
                    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                        analysis = self._analyze_video_file(input_source, source_label)
                    else:
                        analysis = self._analyze_image_parts([input_source], source_label)
                else:
                    raise FileNotFoundError(f"Media path not found: {input_source}")

                result = {
                    "agent": "Face Consistency Agent",
                    "status": "COMPLETED",
                    **analysis,
                }
            except Exception as exc:
                result = {
                    "agent": "Face Consistency Agent",
                    "status": "ERROR",
                    "anomaly_score": 0.0,
                    "details": f"Visual analysis failed: {exc}",
                    "signals": [],
                }

        db_manager.log_agent_execution(
            session_id=session_id,
            agent_name=result["agent"],
            anomaly_score=result["anomaly_score"],
            status=result["status"],
            details=result["details"],
        )
        return result
