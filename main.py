import argparse
import json
import logging
import os
import uuid

from database.clickhouse_db import db_manager
from agents.audio_agent import AudioManipulationAgent
from agents.context_agent import ContextVerificationAgent
from agents.face_agent import FaceConsistencyAgent
from agents.frame_agent import FrameAgent
from agents.master_agent import GeminiMasterSynthesizer
from agents.metadata_agent import MetadataAgent
from config import Config
from database.clickhouse_db import db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Existing project orchestrator: media -> agents -> master verdict."""

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    def __init__(self):
        self.frame_agent = FrameAgent(seconds_interval=2, max_frames=8)
        self.metadata_agent = MetadataAgent()
        self.face_agent = FaceConsistencyAgent()
        self.audio_agent = AudioManipulationAgent()
        self.context_agent = ContextVerificationAgent()
        self.master_agent = GeminiMasterSynthesizer()

    def execute(self, media_path: str, source_label: str | None = None, session_id: str | None = None) -> dict:
        if not media_path or not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        source_label = source_label or os.path.basename(media_path)
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        scan_id = str(uuid.uuid4())
        ext = os.path.splitext(media_path)[1].lower()
        media_type = "video" if ext in self.VIDEO_EXTENSIONS else "image"

        logger.info("Starting CineTruth pipeline for %s", source_label)
        metadata = self.metadata_agent.extract_metadata(media_path)

        frame_paths = []
        if media_type == "video":
            frame_dir = os.path.join(Config.TEMP_DIR, f"frames_{session_id}")
            frame_paths = self.frame_agent.extract_frames(media_path, output_dir=frame_dir)

        # Free-tier/local testing mode: ONE Gemini request returns the three
        # logical agent findings. Local metadata/frame extraction still runs.
        if Config.GEMINI_PIPELINE_MODE != "full":
            bundled = self.master_agent.analyze_media_bundle(
                media_path=media_path,
                source_label=source_label,
                metadata=metadata,
                media_type=media_type,
            )
            agent_outputs = bundled["agents"]
            final_verdict = bundled["final_verdict"]

            # Preserve per-agent telemetry even though efficient mode uses one
            # shared Gemini model request. ClickHouse remains a safe no-op when
            # it is not configured.
            for agent_result in agent_outputs.values():
                score = agent_result.get("anomaly_score", agent_result.get("risk_score", 0.0))
                db_manager.log_agent_execution(
                    session_id=session_id,
                    agent_name=agent_result.get("agent", "Unknown Agent"),
                    anomaly_score=score or 0.0,
                    status=agent_result.get("status", "UNKNOWN"),
                    details=agent_result.get("details", ""),
                )
        else:
            # Full multi-agent mode keeps the original independent model calls.
            # Use this only when the API project has sufficient quota.
            face_result = self.face_agent.analyze_faces(
                session_id=session_id,
                input_source=media_path,
                frame_paths=frame_paths or None,
                source_label=source_label,
            )

            audio_result = self.audio_agent.analyze_audio(
                session_id=session_id,
                input_source=media_path,
                source_label=source_label,
            )

            evidence_summary = json.dumps(
                {
                    "source": source_label,
                    "media_type": media_type,
                    "metadata": metadata,
                    "visual_finding": face_result,
                    "audio_finding": audio_result,
                },
                indent=2,
                default=str,
            )
            context_result = self.context_agent.verify_context(session_id, evidence_summary)

            agent_outputs = {
                "face_agent": face_result,
                "audio_agent": audio_result,
                "context_agent": context_result,
            }
            # master synthesis is local now, so even full mode saves one request.
            final_verdict = self.master_agent.synthesize_verdict(agent_outputs)
            statuses = [str(item.get("status", "ERROR")) for item in agent_outputs.values()]
            if any(status == "COMPLETED" for status in statuses):
                final_verdict["status"] = "COMPLETED"
            elif any(status == "QUOTA_EXCEEDED" for status in statuses):
                final_verdict["status"] = "QUOTA_EXCEEDED"
            elif any(status == "TEMPORARILY_UNAVAILABLE" for status in statuses):
                final_verdict["status"] = "TEMPORARILY_UNAVAILABLE"
            elif any(status == "AUTH_ERROR" for status in statuses):
                final_verdict["status"] = "AUTH_ERROR"
            else:
                final_verdict["status"] = "ERROR"

            final_verdict["gemini_requests_used"] = sum(
                int(item.get("gemini_requests_used", 0) or 0) for item in agent_outputs.values()
            )
            used_models = [
                item.get("gemini_model_used")
                for item in agent_outputs.values()
                if item.get("gemini_model_used")
            ]
            final_verdict["gemini_model_used"] = used_models[0] if used_models else None
            tried_models = []
            for item in agent_outputs.values():
                for model in item.get("gemini_models_tried", []) or []:
                    if model not in tried_models:
                        tried_models.append(model)
            final_verdict["gemini_models_tried"] = tried_models

        # Save scan result to ClickHouse
        try:
            risk_score = float(
                final_verdict.get("overall_manipulation_risk", 0)
            )

            if risk_score >= 70:
                verdict = "DEEPFAKE"
            elif risk_score >= 40:
                verdict = "MANIPULATED"
            else:
                verdict = "AUTHENTIC"

            input_type = (
                "URL"
                if source_label.startswith(("http://", "https://"))
                else "FILE_UPLOAD"
            )

            db_manager.save_scan(
                scan_id=scan_id,
                input_type=input_type,
                source_path=source_label,
                media_type=media_type,
                verdict=verdict,
                confidence_score=risk_score / 100,
            )

        except Exception as exc:
            logger.warning(f"Could not save scan to ClickHouse: {exc}")

        return {
            "scan_id": scan_id,
            "session_id": session_id,
            "source": source_label,
            "media_path": media_path,
            "media_type": media_type,
            "metadata": metadata,
            "frames_sampled": len(frame_paths),
            "frame_paths": frame_paths,
            "pipeline_mode": Config.GEMINI_PIPELINE_MODE,
            "agents": agent_outputs,
            "final_verdict": final_verdict,
        }


def _cli():
    parser = argparse.ArgumentParser(description="Run the CineTruth forensic pipeline on a local image/video.")
    parser.add_argument("media", help="Path to local image or video")
    args = parser.parse_args()

    result = Pipeline().execute(args.media)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _cli()
