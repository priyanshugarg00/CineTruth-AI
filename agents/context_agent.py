import json

from google import genai

from config import Config
from utils.gemini_client import GeminiRequestFailure, generate_content_with_fallback
from database.clickhouse_db import db_manager


class ContextVerificationAgent:
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

    def verify_context(self, session_id: str, evidence_summary: str) -> dict:
        if not self.client:
            result = {
                "agent": "Context Verification Agent",
                "status": "ERROR",
                "risk_score": 0.0,
                "details": "GEMINI_API_KEY is not configured.",
                "gemini_requests_used": 0,
                "gemini_models_tried": [],
            }
        else:
            prompt = f"""
You are CineTruth AI's context verification agent.
Review the following evidence produced by other forensic modules. Judge whether the evidence is internally consistent,
whether there are contradictions, and whether the media warrants additional verification. Do not invent external facts and
do not claim that manipulation is proven solely from these notes.

EVIDENCE:
{evidence_summary}

Return ONLY valid JSON:
{{
  "risk_score": 0.0,
  "summary": "Concise contextual consistency finding"
}}
`risk_score` must be between 0 and 1 and represents contextual verification risk.
"""
            try:
                response, gemini_meta = generate_content_with_fallback(
                    self.client,
                    contents=prompt,
                    primary_model=Config.GEMINI_MODEL,
                )
                data = self._parse_json(response.text)
                risk_score = max(0.0, min(1.0, float(data.get("risk_score", 0.0))))
                details = str(data.get("summary", "Context verification completed."))
                result = {
                    "agent": "Context Verification Agent",
                    "status": "COMPLETED",
                    "risk_score": risk_score,
                    "details": details,
                    "gemini_requests_used": gemini_meta.requests_used,
                    "gemini_model_used": gemini_meta.model_used,
                    "gemini_models_tried": gemini_meta.models_tried,
                }
            except GeminiRequestFailure as exc:
                result = {
                    "agent": "Context Verification Agent",
                    "status": exc.kind,
                    "risk_score": 0.0,
                    "details": exc.public_message,
                    "gemini_requests_used": exc.requests_used,
                    "gemini_models_tried": exc.models_tried,
                    "technical_error": exc.technical_error,
                }
            except Exception as exc:
                result = {
                    "agent": "Context Verification Agent",
                    "status": "ERROR",
                    "risk_score": 0.0,
                    "details": f"Context verification failed: {exc}",
                    "gemini_requests_used": 0,
                    "gemini_models_tried": [],
                }

        db_manager.log_agent_execution(
            session_id=session_id,
            agent_name=result["agent"],
            anomaly_score=result["risk_score"],
            status=result["status"],
            details=result["details"],
        )
        return result
