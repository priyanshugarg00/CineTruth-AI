import json
import mimetypes
import os
import time

from google import genai
from google.genai import types

from config import Config


class GeminiMasterSynthesizer:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

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

    @staticmethod
    def _is_daily_quota_error(exc: Exception) -> bool:
        message = str(exc)
        markers = (
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "generate_content_free_tier_requests",
            "quota_exceeded",
        )
        return any(marker.lower() in message.lower() for marker in markers)

    @staticmethod
    def _score(value, default=0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _signals(value) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item) for item in value[:8]]

    def _wait_for_file(self, uploaded):
        for _ in range(45):
            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", str(state or ""))
            if state_name != "PROCESSING":
                return uploaded
            time.sleep(1)
            uploaded = self.client.files.get(name=uploaded.name)
        return uploaded

    def _media_content(self, media_path: str):
        ext = os.path.splitext(media_path)[1].lower()
        if ext in self.VIDEO_EXTENSIONS:
            uploaded = self.client.files.upload(file=media_path)
            return self._wait_for_file(uploaded)

        mime_type = mimetypes.guess_type(media_path)[0] or "image/jpeg"
        with open(media_path, "rb") as fh:
            return types.Part.from_bytes(data=fh.read(), mime_type=mime_type)

    def analyze_media_bundle(
        self,
        media_path: str,
        source_label: str,
        metadata: dict,
        media_type: str,
    ) -> dict:
        """
        Quota-efficient test path.

        One Gemini generate_content request returns the logical Visual, Audio/AV,
        Context and executive-summary outputs. This keeps the UI/agent result
        structure intact while avoiding four model requests for every scan.
        """
        if not self.client:
            return self._unavailable_bundle("GEMINI_API_KEY is not configured.", media_type)

        try:
            media = self._media_content(media_path)

            audio_instruction = (
                "Analyze audio quality, voice-synthesis indicators, cadence, discontinuities and lip-sync/AV consistency."
                if media_type == "video"
                else "This is a static image. Return audio.status as SKIPPED and do not invent audio evidence."
            )

            prompt = f"""
You are CineTruth AI's quota-efficient multimodal forensic coordinator.
Analyze the supplied {media_type} from: {source_label}

Local metadata (not proof of authenticity):
{json.dumps(metadata, indent=2, default=str)}

Perform THREE distinct roles in this ONE request:
1) VISUAL FORENSICS: inspect face boundaries, geometry, lighting/reflections, texture, warping, duplicated regions,
   temporal consistency where applicable, and obvious synthetic-media indicators.
2) AUDIO/AV FORENSICS: {audio_instruction}
3) CONTEXT CONSISTENCY: judge whether the visible/audible content and metadata are internally consistent and whether
   further verification is warranted. Do not invent external facts.

Do not claim that manipulation is proven. Scores are screening-risk indicators only.
Return ONLY valid JSON in exactly this shape:
{{
  "visual": {{
    "anomaly_score": 0.0,
    "details": "concise visual finding",
    "signals": ["signal 1"]
  }},
  "audio": {{
    "status": "COMPLETED",
    "anomaly_score": 0.0,
    "details": "concise audio/AV finding",
    "signals": ["signal 1"]
  }},
  "context": {{
    "risk_score": 0.0,
    "details": "concise context-consistency finding"
  }},
  "executive_summary": "Two concise calibrated sentences."
}}
All numeric scores must be between 0 and 1.
For an image, audio.status MUST be "SKIPPED" and audio.anomaly_score MUST be 0.
"""

            response = self.client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=[media, prompt],
            )
            data = self._parse_json(response.text)

            visual = data.get("visual") or {}
            audio = data.get("audio") or {}
            context = data.get("context") or {}

            face_result = {
                "agent": "Face Consistency Agent",
                "status": "COMPLETED",
                "anomaly_score": self._score(visual.get("anomaly_score")),
                "details": str(visual.get("details") or "Visual analysis completed."),
                "signals": self._signals(visual.get("signals")),
            }

            if media_type == "image":
                audio_result = {
                    "agent": "Audio Manipulation Agent",
                    "status": "SKIPPED",
                    "anomaly_score": 0.0,
                    "details": "Static image source: no audio track to analyze.",
                    "signals": [],
                }
            else:
                audio_status = str(audio.get("status") or "COMPLETED").upper()
                if audio_status not in {"COMPLETED", "SKIPPED"}:
                    audio_status = "COMPLETED"
                audio_result = {
                    "agent": "Audio Manipulation Agent",
                    "status": audio_status,
                    "anomaly_score": self._score(audio.get("anomaly_score")),
                    "details": str(audio.get("details") or "Audio/AV analysis completed."),
                    "signals": self._signals(audio.get("signals")),
                }

            context_result = {
                "agent": "Context Verification Agent",
                "status": "COMPLETED",
                "risk_score": self._score(context.get("risk_score")),
                "details": str(context.get("details") or "Context verification completed."),
            }

            agents = {
                "face_agent": face_result,
                "audio_agent": audio_result,
                "context_agent": context_result,
            }
            verdict = self.synthesize_verdict(agents, model_summary=str(data.get("executive_summary") or ""))
            verdict["status"] = "COMPLETED"
            verdict["gemini_requests_used"] = 1
            return {"agents": agents, "final_verdict": verdict}

        except Exception as exc:
            if self._is_daily_quota_error(exc):
                return self._quota_bundle(str(exc), media_type)
            return self._unavailable_bundle(f"Gemini analysis failed: {exc}", media_type)

    def _quota_bundle(self, raw_error: str, media_type: str) -> dict:
        message = (
            "Gemini free-tier daily request quota is exhausted for the configured model. "
            "No forensic score was generated. Wait for the daily quota reset or use a project with available quota/billing."
        )
        agents = self._error_agents(message, media_type, status="QUOTA_EXCEEDED")
        return {
            "agents": agents,
            "final_verdict": {
                "status": "QUOTA_EXCEEDED",
                "overall_manipulation_risk": None,
                "executive_summary": message,
                "sub_agent_reports": agents,
                "gemini_requests_used": 0,
                "technical_error": raw_error,
            },
        }

    def _unavailable_bundle(self, message: str, media_type: str) -> dict:
        agents = self._error_agents(message, media_type, status="ERROR")
        return {
            "agents": agents,
            "final_verdict": {
                "status": "ERROR",
                "overall_manipulation_risk": None,
                "executive_summary": message,
                "sub_agent_reports": agents,
                "gemini_requests_used": 0,
            },
        }

    @staticmethod
    def _error_agents(message: str, media_type: str, status: str) -> dict:
        audio_status = "SKIPPED" if media_type == "image" else status
        audio_details = "Static image source: no audio track to analyze." if media_type == "image" else message
        return {
            "face_agent": {
                "agent": "Face Consistency Agent",
                "status": status,
                "anomaly_score": 0.0,
                "details": message,
                "signals": [],
            },
            "audio_agent": {
                "agent": "Audio Manipulation Agent",
                "status": audio_status,
                "anomaly_score": 0.0,
                "details": audio_details,
                "signals": [],
            },
            "context_agent": {
                "agent": "Context Verification Agent",
                "status": status,
                "risk_score": 0.0,
                "details": message,
            },
        }

    def synthesize_verdict(self, agent_outputs: dict, model_summary: str = "") -> dict:
        """Local synthesis: does not spend another Gemini request."""
        weighted = []

        face = agent_outputs.get("face_agent", {})
        if face.get("status") == "COMPLETED":
            weighted.append((self._score(face.get("anomaly_score")), 0.45))

        audio = agent_outputs.get("audio_agent", {})
        if audio.get("status") == "COMPLETED":
            weighted.append((self._score(audio.get("anomaly_score")), 0.35))

        context = agent_outputs.get("context_agent", {})
        if context.get("status") == "COMPLETED":
            weighted.append((self._score(context.get("risk_score")), 0.20))

        if weighted:
            weight_total = sum(weight for _, weight in weighted)
            aggregate = sum(score * weight for score, weight in weighted) / weight_total
            aggregate_risk = max(0, min(100, int(round(aggregate * 100))))
        else:
            aggregate_risk = None

        summary = (model_summary or "").strip()
        if not summary:
            completed = sum(1 for value in agent_outputs.values() if value.get("status") == "COMPLETED")
            if aggregate_risk is None:
                summary = "No reliable forensic score could be generated because the analysis modules did not complete."
            else:
                summary = (
                    f"Manipulation-indicator risk is {aggregate_risk}% based on {completed} completed forensic module(s). "
                    "Treat this as a screening signal and verify suspicious media with additional evidence."
                )

        visual_match = None
        if face.get("status") == "COMPLETED":
            visual_match = int(round((1.0 - self._score(face.get("anomaly_score"))) * 100))

        audio_match = None
        if audio.get("status") == "COMPLETED":
            audio_match = int(round((1.0 - self._score(audio.get("anomaly_score"))) * 100))

        context_match = None
        if context.get("status") == "COMPLETED":
            context_match = int(round((1.0 - self._score(context.get("risk_score"))) * 100))

        return {
            "overall_manipulation_risk": aggregate_risk,
            "executive_summary": summary,
            "sub_agent_reports": agent_outputs,
            "component_match_percentages": {
                "visual": visual_match,
                "audio_av": audio_match,
                "context": context_match,
            },
        }
