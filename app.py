import os
import shutil
import tempfile
import uuid

import streamlit as st
import plotly.graph_objects as go

from config import Config
from database.clickhouse_db import db_manager
from main import Pipeline
from agents.media_storage import media_storage
from agents.takedown_agent import takedown_agent
from utils.report_generator import generate_report

st.set_page_config(
    page_title="CineTruth AI — Forensic Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


def _build_report_bytes(result: dict) -> bytes:
    """Build a PDF while the temporary processing copy still exists."""
    fd, report_path = tempfile.mkstemp(prefix="cinetruth_report_", suffix=".pdf")
    os.close(fd)
    try:
        generate_report(result, report_path)
        with open(report_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(report_path)
        except OSError:
            pass


def _cleanup_processing_artifacts(result: dict):
    """Remove transient frame files/directories created for a scan."""
    frame_paths = list(result.get("frame_paths") or [])
    parent_dirs = set()
    for frame_path in frame_paths:
        if not frame_path:
            continue
        parent_dirs.add(os.path.dirname(frame_path))
        try:
            os.remove(frame_path)
        except OSError:
            pass
    for folder in parent_dirs:
        if folder and os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
            except OSError:
                pass


def _component_match_scores(result: dict) -> tuple[int | None, int | None]:
    """Return visual and audio/AV consistency percentages for display.

    Match/consistency is the inverse of the corresponding anomaly indicator.
    It is intentionally NOT described as biometric identity matching.
    """
    verdict = result.get("final_verdict") or {}
    stored = verdict.get("component_match_percentages") or {}
    visual = stored.get("visual")
    audio = stored.get("audio_av")

    agents = result.get("agents") or {}
    if visual is None:
        face = agents.get("face_agent") or {}
        if face.get("status") == "COMPLETED":
            try:
                anomaly = max(0.0, min(1.0, float(face.get("anomaly_score", 0.0))))
                visual = int(round((1.0 - anomaly) * 100))
            except (TypeError, ValueError):
                visual = None

    if audio is None:
        audio_agent = agents.get("audio_agent") or {}
        if audio_agent.get("status") == "COMPLETED":
            try:
                anomaly = max(0.0, min(1.0, float(audio_agent.get("anomaly_score", 0.0))))
                audio = int(round((1.0 - anomaly) * 100))
            except (TypeError, ValueError):
                audio = None

    return visual, audio


def _render_match_graph(result: dict):
    visual_match, audio_match = _component_match_scores(result)

    st.markdown("#### Image & Audio Match / Consistency")
    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.metric(
            "Image / Visual Match",
            "N/A" if visual_match is None else f"{visual_match}%",
            help="Inverse of the visual anomaly indicator. Higher means the visual evidence appeared more internally consistent.",
        )
    with metric_b:
        st.metric(
            "Audio / AV Match",
            "N/A" if audio_match is None else f"{audio_match}%",
            help="Inverse of the audio/AV anomaly indicator. Static images show N/A because no audio is analyzed.",
        )

    labels = []
    values = []
    if visual_match is not None:
        labels.append("Image / Visual")
        values.append(visual_match)
    if audio_match is not None:
        labels.append("Audio / AV")
        values.append(audio_match)

    if values:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{value}%" for value in values],
                    textposition="outside",
                    hovertemplate="%{x}: %{y}%<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title="Match / consistency (%)", range=[0, 105]),
            xaxis=dict(title="Forensic component"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"match_graph_{result.get('session_id', id(result))}")

    st.caption(
        "Match % is a forensic consistency indicator calculated as 100 - anomaly %. "
        "It is not biometric identity matching and is not proof of authenticity."
    )


load_css("assets/style.css")
pipeline = Pipeline()

st.markdown(
    """
<div class="main-header">
    <div class="main-title">🛡️ CineTruth AI & Rights Protect</div>
    <div class="main-subtitle">Autonomous Deepfake Detection, Identity Protection & Legal Takedown Suite</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ System Telemetry")
    gemini_status = "🟢 Configured" if Config.GEMINI_API_KEY else "🔴 Missing API Key"
    clickhouse_status = "🟢 Connected" if db_manager.client else ("🟡 Not configured" if not db_manager.configured else "🔴 Connection failed")
    serp_status = "🟢 Configured" if Config.SERP_API_KEY else "🔴 Missing"
    imgbb_status = "🟢 Configured" if Config.IMGBB_API_KEY else "🔴 Missing"
    r2_status = "🟢 Configured" if media_storage.configured else "🔴 Missing configuration"

    st.caption(f"**Gemini:** {gemini_status}")
    st.caption(f"**Gemini pipeline:** `{Config.GEMINI_PIPELINE_MODE}`")
    st.caption(f"**SerpAPI:** {serp_status}")
    st.caption(f"**ImgBB:** {imgbb_status}")
    st.caption(f"**Cloudflare R2:** {r2_status}")
    st.caption(f"**ClickHouse:** {clickhouse_status}")
    st.divider()
    st.info("ClickHouse is optional during local functional testing.")


tab1, tab2 = st.tabs(["🎬 Deepfake Media Detection", "👤 Identity Shield & Auto-Web Takedown"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    # Batch items use the same existing single-media Pipeline.execute() method.
    # This keeps all forensic logic unchanged while allowing multiple inputs.
    batch_media_items = []

    with col1:
        st.subheader("📁 Media Input Workspace")
        input_type = st.radio(
            "Choose Input Type:",
            ["Upload Files (Video/Image)", "Media URLs"],
            horizontal=True,
        )

        if input_type == "Upload Files (Video/Image)":
            uploaded_files = st.file_uploader(
                "Upload one or more media files (MP4, MOV, JPG, PNG)",
                type=["mp4", "mov", "webm", "avi", "mkv", "jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="forensic_batch_uploads",
            )

            if uploaded_files and not media_storage.configured:
                st.error(
                    "Cloudflare R2 is not configured. Add R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
                    "R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME to your .env file."
                )

            if uploaded_files and media_storage.configured:
                st.caption(f"{len(uploaded_files)} file(s) selected")
                upload_cache = st.session_state.setdefault("r2_upload_cache", {})

                for idx, uploaded_file in enumerate(uploaded_files, start=1):
                    try:
                        file_id = getattr(uploaded_file, "file_id", None)
                        cache_key = str(file_id or f"{uploaded_file.name}:{uploaded_file.size}")
                        stored = upload_cache.get(cache_key)
                        if not stored:
                            with st.spinner(f"Uploading {uploaded_file.name} to Cloudflare R2..."):
                                stored = media_storage.upload_streamlit_file(uploaded_file)
                            upload_cache[cache_key] = stored

                        item = {
                            "source_label": uploaded_file.name,
                            "storage_key": stored["key"],
                            "storage_record": stored,
                            "input_kind": "upload",
                        }
                        batch_media_items.append(item)

                        with st.expander(f"Preview #{idx}: {uploaded_file.name}", expanded=(idx == 1)):
                            preview_url = media_storage.get_access_url(stored["key"])
                            if uploaded_file.type and uploaded_file.type.startswith("video"):
                                st.video(preview_url)
                            else:
                                st.image(preview_url, caption=uploaded_file.name, use_container_width=True)
                            st.caption(
                                f"Stored in R2 · {stored['size']/(1024*1024):.2f} MB · "
                                f"`{stored['key']}`"
                            )
                    except Exception as exc:
                        st.error(f"Could not upload `{uploaded_file.name}` to R2: {exc}")

        else:
            urls_text = st.text_area(
                "Enter one media/video-page URL per line:",
                placeholder=(
                    "https://example.com/video.mp4\n"
                    "https://www.youtube.com/watch?v=...\n"
                    "https://example.com/image.jpg"
                ),
                height=140,
                key="forensic_urls_text",
            )

            parsed_urls = []
            seen_urls = set()
            for line in urls_text.splitlines():
                candidate = line.strip()
                if candidate and candidate not in seen_urls:
                    parsed_urls.append(candidate)
                    seen_urls.add(candidate)

            if parsed_urls:
                st.caption(f"{len(parsed_urls)} unique URL(s) entered")

            if st.button(
                "☁️ Resolve & Store All URL Media in R2",
                use_container_width=True,
                disabled=not parsed_urls or not media_storage.configured,
                key="load_batch_urls",
            ):
                loaded_items = []
                load_errors = []
                progress = st.progress(0, text="Resolving URLs and uploading media to R2...")

                for idx, media_url in enumerate(parsed_urls, start=1):
                    try:
                        stored = media_storage.ingest_url(media_url)
                        loaded_items.append(
                            {
                                "source_label": media_url,
                                "storage_key": stored["key"],
                                "storage_record": stored,
                                "input_kind": "url",
                            }
                        )
                    except Exception as exc:
                        load_errors.append({"source": media_url, "error": str(exc)})
                    finally:
                        progress.progress(
                            idx / max(len(parsed_urls), 1),
                            text=f"Resolving/uploading URL {idx}/{len(parsed_urls)}",
                        )

                progress.empty()
                st.session_state["url_media_items"] = loaded_items
                st.session_state["url_media_errors"] = load_errors

            if parsed_urls and not media_storage.configured:
                st.error("Configure Cloudflare R2 before loading media URLs.")

            stored_url_items = st.session_state.get("url_media_items", [])
            stored_url_errors = st.session_state.get("url_media_errors", [])

            for item in stored_url_items:
                if item.get("storage_key"):
                    batch_media_items.append(item)

            if stored_url_items:
                st.success(f"Stored {len(batch_media_items)} URL media item(s) in Cloudflare R2.")
                for idx, item in enumerate(batch_media_items, start=1):
                    stored = item.get("storage_record") or {}
                    key = item["storage_key"]
                    ext = os.path.splitext(stored.get("filename") or key)[1].lower()
                    with st.expander(f"URL Media #{idx}", expanded=(idx == 1)):
                        st.caption(item["source_label"])
                        preview_url = media_storage.get_access_url(key)
                        if ext in {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}:
                            st.video(preview_url)
                        else:
                            st.image(preview_url, caption="R2 Stored Image", use_container_width=True)
                        st.caption(f"Cloudflare R2 key: `{key}`")

            if stored_url_errors:
                with st.expander(f"⚠️ URL load errors ({len(stored_url_errors)})"):
                    for err in stored_url_errors:
                        st.error(f"{err['source']}\n\n{err['error']}")

        if batch_media_items:
            st.info(
                f"Ready to analyze **{len(batch_media_items)}** media item(s). "
                "Items are processed sequentially so one failed item does not stop the whole batch."
            )
            if Config.GEMINI_PIPELINE_MODE != "full":
                st.caption(
                    f"Efficient mode estimate: about {len(batch_media_items)} Gemini request(s) "
                    f"for this batch (approximately 1 per media item)."
                )

    with col2:
        st.subheader("📊 Manipulation Verdict & Multi-Agent Telemetry")

        if batch_media_items:
            button_label = (
                "🚀 Run Deepfake Scan Pipeline"
                if len(batch_media_items) == 1
                else f"🚀 Run Batch Deepfake Scan ({len(batch_media_items)} items)"
            )

            if st.button(button_label, type="primary", use_container_width=True, key="run_forensic_batch"):
                results = []
                errors = []
                progress = st.progress(0, text="Starting forensic batch...")

                for idx, item in enumerate(batch_media_items, start=1):
                    source_label = item["source_label"]
                    storage_key = item["storage_key"]
                    storage_record = item.get("storage_record") or {}
                    session_id = f"session_{uuid.uuid4().hex[:8]}"

                    try:
                        progress.progress(
                            (idx - 1) / max(len(batch_media_items), 1),
                            text=f"Analyzing {idx}/{len(batch_media_items)}: {source_label}",
                        )

                        # OpenCV/Gemini currently require a filesystem path, so R2 media is
                        # downloaded to an OS temporary file only for this analysis. It is
                        # deleted automatically when this block exits.
                        with media_storage.local_copy(storage_key) as local_media_path:
                            result = pipeline.execute(
                                local_media_path,
                                source_label=source_label,
                                session_id=session_id,
                            )
                            result["batch_index"] = idx
                            result["storage"] = {
                                "provider": "cloudflare_r2",
                                "bucket": Config.R2_BUCKET_NAME,
                                "key": storage_key,
                                "sha256": storage_record.get("sha256"),
                                "size": storage_record.get("size"),
                            }

                            # Build the PDF before temporary media/frame files are removed.
                            # A report failure must not discard an otherwise successful scan.
                            try:
                                result["_report_pdf_bytes"] = _build_report_bytes(result)
                            except Exception as report_exc:
                                result["_report_pdf_bytes"] = None
                                result["_report_error"] = str(report_exc)
                            finally:
                                _cleanup_processing_artifacts(result)

                        # Never persist the OS temporary path in session state/results.
                        result["media_path"] = f"r2://{Config.R2_BUCKET_NAME}/{storage_key}"
                        result["frame_paths"] = []
                        results.append(result)
                    except Exception as exc:
                        errors.append(
                            {
                                "batch_index": idx,
                                "source": source_label,
                                "error": str(exc),
                            }
                        )
                    finally:
                        progress.progress(
                            idx / max(len(batch_media_items), 1),
                            text=f"Processed {idx}/{len(batch_media_items)}",
                        )

                progress.empty()
                st.session_state["last_forensic_results"] = results
                st.session_state["last_forensic_batch_errors"] = errors
                # Keep backward compatibility for any code that still expects one last result.
                if results:
                    st.session_state["last_forensic_result"] = results[-1]

                if results:
                    st.success(f"Completed {len(results)} of {len(batch_media_items)} media item(s).")
                if errors:
                    st.warning(f"{len(errors)} item(s) failed. Other items were still processed.")

        results = st.session_state.get("last_forensic_results", [])
        batch_errors = st.session_state.get("last_forensic_batch_errors", [])

        if results:
            st.markdown(f"### Batch Results ({len(results)})")

            for result_pos, result in enumerate(results, start=1):
                verdict = result["final_verdict"]
                agents = result["agents"]
                risk_score = verdict.get("overall_manipulation_risk")
                verdict_status = verdict.get("status", "COMPLETED")
                source = result.get("source", f"Media #{result_pos}")
                media_type = result.get("media_type", "media")

                if risk_score is None:
                    title_risk = "No score"
                else:
                    title_risk = f"{risk_score}% risk"

                with st.expander(
                    f"#{result.get('batch_index', result_pos)} · {media_type.title()} · {title_risk} · {source}",
                    expanded=(result_pos == 1),
                ):
                    if verdict_status == "QUOTA_EXCEEDED":
                        st.warning(
                            verdict.get(
                                "executive_summary",
                                "Gemini quota is exhausted. Automatic retries/fallbacks were attempted, but no score was generated.",
                            )
                        )
                    elif verdict_status == "TEMPORARILY_UNAVAILABLE":
                        st.warning(
                            verdict.get(
                                "executive_summary",
                                "Gemini is temporarily busy. Automatic retries and fallback models were attempted. Please retry shortly.",
                            )
                        )
                    elif verdict_status == "AUTH_ERROR":
                        st.error(
                            "Gemini authorization failed. Check GEMINI_API_KEY in your Streamlit secrets and API project permissions."
                        )
                    elif risk_score is None:
                        st.error(verdict.get("executive_summary", "No reliable forensic score could be generated."))
                    else:
                        card_class = "risk-card-high"
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <div style="font-weight:600;color:#f43f5e;font-size:1.1rem;">MANIPULATION INDICATOR RISK</div>
                                <div class="risk-score-text">{risk_score}%</div>
                                <div style="color:#fda4af;font-size:0.9rem;">{verdict.get('executive_summary','')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    gemini_model = verdict.get("gemini_model_used")
                    gemini_attempts = verdict.get("gemini_requests_used", "N/A")
                    gemini_caption = f"Gemini attempts: {gemini_attempts}"
                    if gemini_model:
                        gemini_caption += f" · Model used: {gemini_model}"
                    elif verdict.get("gemini_models_tried"):
                        gemini_caption += " · Models tried: " + ", ".join(verdict.get("gemini_models_tried", []))

                    st.caption(
                        f"Source: {source} · Pipeline mode: {result.get('pipeline_mode', 'unknown')} · "
                        f"{gemini_caption}"
                    )

                    _render_match_graph(result)

                    st.markdown("#### Detailed Findings")
                    face_res = agents.get("face_agent", {})
                    audio_res = agents.get("audio_agent", {})
                    context_res = agents.get("context_agent", {})
                    st.info(f"**Visual [{face_res.get('status','N/A')}]:** {face_res.get('details','N/A')}")
                    st.info(f"**Audio/AV [{audio_res.get('status','N/A')}]:** {audio_res.get('details','N/A')}")
                    st.info(f"**Context [{context_res.get('status','N/A')}]:** {context_res.get('details','N/A')}")

                    report_bytes = result.get("_report_pdf_bytes")
                    if result.get("_report_error"):
                        st.warning(f"PDF report could not be generated: {result['_report_error']}")
                    if report_bytes:
                        st.download_button(
                            "📥 Download Forensic Report (PDF)",
                            data=report_bytes,
                            file_name=f"CineTruth_Forensic_{result['session_id']}.pdf",
                            mime="application/pdf",
                            key=f"report_{result['session_id']}",
                        )

                    storage_info = result.get("storage") or {}
                    if storage_info.get("key"):
                        st.caption(
                            f"Permanent media storage: Cloudflare R2 · "
                            f"`{storage_info.get('bucket')}/{storage_info.get('key')}`"
                        )

        if batch_errors:
            with st.expander(f"❌ Batch analysis errors ({len(batch_errors)})"):
                for err in batch_errors:
                    st.error(f"#{err['batch_index']} — {err['source']}\n\n{err['error']}")

        if not batch_media_items and not results:
            st.caption("Awaiting one or more media files or URLs.")


with tab2:
    st.subheader("🔎 Identity Matching & Reverse Web-Search Suite")
    st.caption("Upload a reference photo or use a public photo URL to run Google Lens via SerpAPI.")

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown("### Step 1: Reference Image")
        id_input_type = st.radio(
            "Reference Photo Source:",
            ["Upload Local Photo", "Photo URL"],
            horizontal=True,
        )
        ref_photo_loaded = False
        ref_payload = None

        if id_input_type == "Upload Local Photo":
            ref_image = st.file_uploader("Upload Reference Face Photo", type=["jpg", "png", "jpeg", "webp"], key="identity_file")
            if ref_image:
                st.image(ref_image, caption="Reference Identity Loaded", width=200)
                ref_photo_loaded = True
                ref_payload = ref_image
        else:
            ref_image_url = st.text_input("Enter Reference Photo URL:", placeholder="https://example.com/photo.jpg")
            if ref_image_url:
                st.image(ref_image_url, caption="Reference Image Loaded from Link", width=200)
                ref_photo_loaded = True
                ref_payload = ref_image_url

        if ref_photo_loaded and st.button("🔍 Run Autonomous Web Reverse-Search", type="primary", use_container_width=True):
            with st.spinner("Running Google Lens visual search through SerpAPI..."):
                matches = takedown_agent.search_unauthorized_matches(ref_payload)
                st.session_state["discovered_matches"] = matches
            if matches:
                st.success(f"Found {len(matches)} visual-search candidate(s).")
            else:
                st.warning("No visual-search candidates returned. Check SERP_API_KEY/IMGBB_API_KEY and the input image.")

    with col_b:
        st.markdown("### Step 2: Search Candidates & Review Notices")
        matches = st.session_state.get("discovered_matches", [])
        if matches:
            for idx, item in enumerate(matches):
                with st.expander(f"Candidate #{idx+1} — {item.get('platform','Web')}", expanded=(idx == 0)):
                    st.markdown(f"**URL:** `{item.get('target_url','')}`")
                    st.write(f"**Status:** {item.get('status','')}")
                    st.caption("Google Lens candidate ≠ verified biometric identity match. Human verification is required.")
                    if item.get("thumbnail"):
                        st.image(item["thumbnail"], width=120)

                    portal_type = st.selectbox(
                        f"Notice Type (Candidate #{idx+1}):",
                        ["Platform Content Review Request", "Copyright/DMCA Review Draft", "Cyber Crime Incident Draft"],
                        key=f"portal_{idx}",
                    )
                    if st.button(f"⚖️ Generate Draft for Candidate #{idx+1}", key=f"gen_btn_{idx}"):
                        nd = takedown_agent.generate_notice(
                            target_url=item["target_url"],
                            similarity_score=item.get("similarity_score"),
                            notice_type=portal_type,
                        )
                        st.session_state[f"notice_{idx}"] = nd

                    if f"notice_{idx}" in st.session_state:
                        nd = st.session_state[f"notice_{idx}"]
                        st.success(f"Draft generated. Request ID: `{nd['request_id']}`")
                        st.code(nd["notice_body"], language="text")
                        st.download_button(
                            "📥 Download Notice (.txt)",
                            data=nd["notice_body"],
                            file_name=f"Takedown_{nd['request_id']}.txt",
                            mime="text/plain",
                            key=f"dl_{idx}",
                        )
        else:
            st.info("Run a reverse search from the left panel to show live visual-search candidates here.")
