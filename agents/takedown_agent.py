import base64
import os
import uuid
import requests

from config import Config
from google import genai

try:
    from database.clickhouse_db import db_manager
except ImportError:
    db_manager = None


class TakedownAgent:
    def __init__(self):
        self.gemini_api_key = Config.GEMINI_API_KEY
        self.client = genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None
        self.serp_api_key = Config.SERP_API_KEY
        self.imgbb_api_key = Config.IMGBB_API_KEY

    def _upload_to_temp_host(self, image_file_or_bytes) -> str | None:
        if not self.imgbb_api_key:
            print("[TakedownAgent] IMGBB_API_KEY is missing.")
            return None

        try:
            if hasattr(image_file_or_bytes, "read"):
                image_bytes = image_file_or_bytes.read()
                image_file_or_bytes.seek(0)
            elif isinstance(image_file_or_bytes, bytes):
                image_bytes = image_file_or_bytes
            else:
                return None

            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            res = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": self.imgbb_api_key, "image": b64_img},
                timeout=20,
            )
            res.raise_for_status()
            return res.json()["data"]["url"]
        except Exception as exc:
            print(f"[TakedownAgent] Temporary image upload failed: {exc}")
            return None

    def search_unauthorized_matches(self, input_data="User Identity") -> list:
        public_image_url = None
        if isinstance(input_data, str) and input_data.startswith("http"):
            public_image_url = input_data
        elif hasattr(input_data, "read") or isinstance(input_data, bytes):
            public_image_url = self._upload_to_temp_host(input_data)

        if not public_image_url:
            return []
        if not self.serp_api_key:
            print("[TakedownAgent] SERP_API_KEY is missing.")
            return []

        try:
            params = {
                "engine": "google_lens",
                "url": public_image_url,
                "api_key": self.serp_api_key,
            }
            res = requests.get("https://serpapi.com/search", params=params, timeout=30)
            res.raise_for_status()
            data = res.json()
            visual_matches = data.get("visual_matches") or data.get("images_results") or []

            discovered_targets = []
            for match in visual_matches[:8]:
                link = match.get("link") or match.get("source_first_page") or ""
                if not link:
                    continue
                source = match.get("source", "Web result")
                matched_img_src = match.get("thumbnail") or match.get("original") or ""

                platform = source
                lower_link = link.lower()
                if "instagram.com" in lower_link:
                    platform = "Instagram"
                elif "facebook.com" in lower_link:
                    platform = "Facebook"
                elif "reddit.com" in lower_link:
                    platform = "Reddit"
                elif "x.com" in lower_link or "twitter.com" in lower_link:
                    platform = "X (Twitter)"

                # Google Lens does not provide a reliable biometric similarity percentage.
                discovered_targets.append(
                    {
                        "target_url": link,
                        "matched_image_url": matched_img_src,
                        "platform": platform,
                        "title": match.get("title", "Visual search candidate"),
                        "similarity_score": None,
                        "thumbnail": matched_img_src,
                        "status": "VISUAL_SEARCH_CANDIDATE",
                    }
                )
            return discovered_targets
        except Exception as exc:
            print(f"[TakedownAgent] SerpAPI search failed: {exc}")
            return []

    def generate_notice(self, target_url: str, similarity_score=None, notice_type: str = "Content Review Request") -> dict:
        request_id = f"CTR-{uuid.uuid4().hex[:10].upper()}"
        score_text = "not independently verified" if similarity_score is None else f"{similarity_score}%"

        notice_body = ""
        if self.client:
            prompt = f"""
Draft a concise, factual content review/takedown request for a potentially unauthorized use of a person's image.
Do not make unsupported legal conclusions. State that the match is a visual-search candidate and requires human verification.

Request ID: {request_id}
Target URL: {target_url}
Visual similarity score: {score_text}
Requested notice type: {notice_type}

Return plain text only.
"""
            try:
                response = self.client.models.generate_content(model=Config.GEMINI_MODEL, contents=prompt)
                notice_body = (response.text or "").strip()
            except Exception:
                notice_body = ""

        if not notice_body:
            notice_body = f"""CineTruth AI Content Review Request
Request ID: {request_id}
Target URL: {target_url}
Notice type: {notice_type}

I am requesting a human review of the content at the URL above because an automated visual search returned it as a possible match to a reference image. This result is not a verified biometric identity match and does not by itself establish infringement or impersonation.

Please review the content, applicable rights/policies, and any supporting evidence before taking action.
"""

        if db_manager is not None:
            db_manager.log_takedown_request(
                request_id=request_id,
                target_url=target_url,
                similarity_score=float(similarity_score or 0.0),
                notice_type=notice_type,
            )

        return {
            "request_id": request_id,
            "target_url": target_url,
            "notice_type": notice_type,
            "notice_body": notice_body,
        }


takedown_agent = TakedownAgent()
