from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Brand / document palette.
NAVY = colors.HexColor("#0F172A")
BLUE = colors.HexColor("#2563EB")
SLATE = colors.HexColor("#475569")
MUTED = colors.HexColor("#64748B")
BORDER = colors.HexColor("#CBD5E1")
LIGHT = colors.HexColor("#F8FAFC")
GREEN = colors.HexColor("#15803D")
AMBER = colors.HexColor("#B45309")
ORANGE = colors.HexColor("#C2410C")
RED = colors.HexColor("#B91C1C")
SOFT_GREEN = colors.HexColor("#F0FDF4")
SOFT_AMBER = colors.HexColor("#FFFBEB")
SOFT_ORANGE = colors.HexColor("#FFF7ED")
SOFT_RED = colors.HexColor("#FEF2F2")
SOFT_BLUE = colors.HexColor("#EFF6FF")


def _safe_text(value: Any, max_chars: int | None = None) -> str:
    """Return ReportLab-safe readable text using built-in PDF fonts."""
    if value is None:
        text = "N/A"
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)

    text = text.replace("\x1b", "")
    text = re.sub(r"\[[0-9;]*m", "", text)  # strip common ANSI color suffixes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...").replace("\u2022", "-")
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    # Helvetica/WinAnsi cannot render every Unicode glyph. Replace unsupported
    # characters rather than producing black squares in the PDF.
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _p(value: Any, style: ParagraphStyle, max_chars: int | None = None) -> Paragraph:
    text = escape(_safe_text(value, max_chars=max_chars)).replace("\n", "<br/>")
    return Paragraph(text or "N/A", style)


def _sha256(path: str | None) -> str:
    if not path or not os.path.isfile(path):
        return "Unavailable"
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "Unavailable"


def _human_size(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return _safe_text(value)
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _risk_label(score: Any) -> tuple[str, colors.Color, colors.Color]:
    if score is None:
        return "Unavailable", MUTED, LIGHT
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Unavailable", MUTED, LIGHT
    if score < 30:
        return "Low indicator risk", GREEN, SOFT_GREEN
    if score < 60:
        return "Moderate indicator risk", AMBER, SOFT_AMBER
    if score < 80:
        return "High indicator risk", ORANGE, SOFT_ORANGE
    return "Very high indicator risk", RED, SOFT_RED


def _agent_score(agent: dict) -> str:
    value = agent.get("anomaly_score", agent.get("risk_score"))
    if value is None:
        return "N/A"
    try:
        score = float(value)
        if 0 <= score <= 1:
            score *= 100
        return f"{score:.0f}%"
    except (TypeError, ValueError):
        return _safe_text(value)


def _component_match_scores(data: dict) -> tuple[int | None, int | None]:
    verdict = data.get("final_verdict") or {}
    stored = verdict.get("component_match_percentages") or {}
    visual = stored.get("visual")
    audio = stored.get("audio_av")
    agents = data.get("agents") or {}

    def _derive(agent_key: str):
        agent = agents.get(agent_key) or {}
        if agent.get("status") != "COMPLETED":
            return None
        try:
            anomaly = max(0.0, min(1.0, float(agent.get("anomaly_score", 0.0))))
            return int(round((1.0 - anomaly) * 100))
        except (TypeError, ValueError):
            return None

    if visual is None:
        visual = _derive("face_agent")
    if audio is None:
        audio = _derive("audio_agent")

    try:
        visual = None if visual is None else max(0, min(100, int(round(float(visual)))))
    except (TypeError, ValueError):
        visual = None
    try:
        audio = None if audio is None else max(0, min(100, int(round(float(audio)))))
    except (TypeError, ValueError):
        audio = None
    return visual, audio


def _match_bar_chart(visual: int | None, audio: int | None, width: float = 158 * mm):
    rows = [("Image / Visual", visual), ("Audio / AV", audio)]
    height = 38 * mm
    drawing = Drawing(width, height)
    label_x = 0
    bar_x = 34 * mm
    bar_width = width - bar_x - 16 * mm
    bar_height = 6 * mm
    y_positions = [25 * mm, 11 * mm]

    for (label, score), y in zip(rows, y_positions):
        drawing.add(String(label_x, y + 1.4 * mm, label, fontName="Helvetica-Bold", fontSize=8, fillColor=NAVY))
        drawing.add(Rect(bar_x, y, bar_width, bar_height, fillColor=colors.HexColor("#E2E8F0"), strokeColor=None))
        if score is None:
            drawing.add(String(bar_x + 2 * mm, y + 1.4 * mm, "N/A", fontName="Helvetica", fontSize=8, fillColor=MUTED))
        else:
            fill_width = bar_width * (float(score) / 100.0)
            drawing.add(Rect(bar_x, y, fill_width, bar_height, fillColor=BLUE, strokeColor=None))
            drawing.add(String(bar_x + bar_width + 2 * mm, y + 1.4 * mm, f"{score}%", fontName="Helvetica-Bold", fontSize=8, fillColor=NAVY))

    drawing.add(String(bar_x, 2.0 * mm, "0%", fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    drawing.add(String(bar_x + bar_width - 7 * mm, 2.0 * mm, "100%", fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    return drawing


def _preview_path(data: dict) -> str | None:
    media_type = str(data.get("media_type") or "").lower()
    media_path = data.get("media_path")
    if media_type == "image" and media_path and os.path.isfile(media_path):
        return media_path
    for frame in data.get("frame_paths") or []:
        if frame and os.path.isfile(frame):
            return frame
    return None


def _scaled_preview(path: str, max_width: float, max_height: float):
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            width, height = im.size
        if not width or not height:
            return None
        ratio = min(max_width / float(width), max_height / float(height))
        return Image(path, width=width * ratio, height=height * ratio)
    except Exception as exc:
        logger.warning("Could not add media preview to PDF: %s", exc)
        return None


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CTTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "CTSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "CTH1",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
        ),
        "h2": ParagraphStyle(
            "CTH2",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=NAVY,
            spaceBefore=2 * mm,
            spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "CTBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.5,
            textColor=SLATE,
            spaceAfter=1.5 * mm,
        ),
        "small": ParagraphStyle(
            "CTSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
        "label": ParagraphStyle(
            "CTLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=11,
            textColor=NAVY,
        ),
        "value": ParagraphStyle(
            "CTValue",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=SLATE,
        ),
        "table_head": ParagraphStyle(
            "CTTableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "table": ParagraphStyle(
            "CTTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.5,
            textColor=SLATE,
        ),
        "score": ParagraphStyle(
            "CTScore",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=30,
            alignment=TA_CENTER,
        ),
        "score_label": ParagraphStyle(
            "CTScoreLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
        ),
        "center_small": ParagraphStyle(
            "CTCenterSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def generate_report(data: dict, output_filename: str = "CineTruth_Forensic_Report.pdf") -> bool:
    """Generate a polished PDF report from an existing Pipeline result dict.

    The function keeps the previous public name (`generate_report`) so callers
    do not need a new API. The output is now PDF; use a `.pdf` filename.
    """
    logger.info("Generating PDF report: %s", output_filename)
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)

    styles = _styles()
    session_id = _safe_text(data.get("session_id") or "N/A")
    source = data.get("source") or os.path.basename(str(data.get("media_path") or "Unknown source"))
    media_type = _safe_text(data.get("media_type") or "Unknown").title()
    pipeline_mode = _safe_text(data.get("pipeline_mode") or "Unknown")
    verdict = data.get("final_verdict") or {}
    agents = data.get("agents") or {}
    metadata = data.get("metadata") or {}
    risk_score = verdict.get("overall_manipulation_risk")
    risk_label, risk_color, risk_bg = _risk_label(risk_score)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    file_hash = _sha256(data.get("media_path"))

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"CineTruth AI Forensic Screening Report - {session_id}",
        author="CineTruth AI",
        subject="AI-assisted media forensic screening report",
    )

    story = []

    # Header / identity
    story.append(_p("CineTruth AI", styles["title"]))
    story.append(_p("Media Forensic Screening Report", styles["subtitle"]))
    story.append(
        HRFlowable(width="100%", thickness=1.4, color=BLUE, spaceBefore=0, spaceAfter=4 * mm)
    )

    overview_data = [
        [_p("Report ID", styles["label"]), _p(session_id, styles["value"]),
         _p("Generated", styles["label"]), _p(generated_at, styles["value"])],
        [_p("Media type", styles["label"]), _p(media_type, styles["value"]),
         _p("Pipeline mode", styles["label"]), _p(pipeline_mode, styles["value"])],
        [_p("Source", styles["label"]), _p(source, styles["value"], max_chars=500),
         _p("Gemini requests", styles["label"]), _p(verdict.get("gemini_requests_used", "N/A"), styles["value"])],
    ]
    overview = Table(overview_data, colWidths=[24 * mm, 54 * mm, 26 * mm, 54 * mm], hAlign="LEFT")
    overview.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(overview)
    story.append(Spacer(1, 4 * mm))

    # Risk / executive assessment
    story.append(_p("Executive Assessment", styles["h1"]))
    score_text = "N/A" if risk_score is None else f"{risk_score}%"
    score_style = ParagraphStyle("RiskScoreDynamic", parent=styles["score"], textColor=risk_color)
    label_style = ParagraphStyle("RiskLabelDynamic", parent=styles["score_label"], textColor=risk_color)
    summary = verdict.get("executive_summary") or "No executive summary was generated."
    assessment = Table(
        [
            [
                Table([
                    [_p(score_text, score_style)],
                    [_p(risk_label.upper(), label_style)],
                ], colWidths=[45 * mm]),
                _p(summary, styles["body"], max_chars=2500),
            ]
        ],
        colWidths=[49 * mm, 109 * mm],
        hAlign="LEFT",
    )
    assessment.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), risk_bg),
        ("BACKGROUND", (1, 0), (1, 0), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(assessment)
    story.append(Spacer(1, 2.5 * mm))
    story.append(_p(
        "Interpretation: this score is a screening indicator produced from automated forensic signals. "
        "It is not, by itself, proof that media is authentic or manipulated.",
        styles["small"],
    ))

    # Visual/audio consistency percentages and graph.
    visual_match, audio_match = _component_match_scores(data)
    story.append(_p("Image & Audio Match / Consistency", styles["h1"]))
    match_table = Table(
        [[
            _p("Image / Visual Match", styles["label"]),
            _p("N/A" if visual_match is None else f"{visual_match}%", styles["value"]),
            _p("Audio / AV Match", styles["label"]),
            _p("N/A" if audio_match is None else f"{audio_match}%", styles["value"]),
        ]],
        colWidths=[40 * mm, 29 * mm, 40 * mm, 29 * mm],
        hAlign="LEFT",
    )
    match_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(match_table)
    story.append(Spacer(1, 2 * mm))
    story.append(_match_bar_chart(visual_match, audio_match))
    story.append(_p(
        "Match/consistency % is calculated as 100 - the corresponding anomaly indicator. "
        "It describes automated forensic consistency, not biometric identity matching and not proof of authenticity.",
        styles["small"],
    ))

    # Preview evidence
    preview = _preview_path(data)
    if preview:
        preview_flowable = _scaled_preview(preview, 158 * mm, 70 * mm)
        if preview_flowable:
            story.append(_p("Media Preview", styles["h1"]))
            preview_table = Table([[preview_flowable]], colWidths=[158 * mm], hAlign="LEFT")
            preview_table.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(preview_table)
            if str(data.get("media_type")).lower() == "video":
                story.append(_p("Preview shown from the first sampled video frame.", styles["small"]))

    # Agent findings
    story.append(_p("Agent Findings", styles["h1"]))
    agent_rows = [[
        _p("Module", styles["table_head"]),
        _p("Status", styles["table_head"]),
        _p("Risk", styles["table_head"]),
        _p("Finding", styles["table_head"]),
    ]]
    ordered_agents = ["face_agent", "audio_agent", "context_agent"]
    for key in ordered_agents + [k for k in agents.keys() if k not in ordered_agents]:
        agent = agents.get(key) or {}
        if not agent:
            continue
        agent_rows.append([
            _p(agent.get("agent") or key.replace("_", " ").title(), styles["table"]),
            _p(agent.get("status", "N/A"), styles["table"]),
            _p(_agent_score(agent), styles["table"]),
            _p(agent.get("details", "N/A"), styles["table"], max_chars=3000),
        ])

    agent_table = LongTable(agent_rows, colWidths=[36 * mm, 22 * mm, 16 * mm, 84 * mm], repeatRows=1, hAlign="LEFT")
    agent_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(agent_table)

    # Signals as a separate readable section.
    signal_blocks = []
    for key in ordered_agents:
        agent = agents.get(key) or {}
        signals = agent.get("signals") or []
        if not signals:
            continue
        if not isinstance(signals, (list, tuple)):
            signals = [signals]
        signal_blocks.append(_p(agent.get("agent") or key, styles["h2"]))
        for signal in signals[:8]:
            signal_blocks.append(_p(f"- {signal}", styles["body"], max_chars=1000))
    if signal_blocks:
        story.append(Spacer(1, 2 * mm))
        story.append(_p("Observed Signals", styles["h2"]))
        story.extend(signal_blocks)

    # Evidence integrity / processing summary
    story.append(_p("Evidence & Processing Details", styles["h1"]))
    evidence_rows = [
        [_p("SHA-256 media fingerprint", styles["label"]), _p(file_hash, styles["value"])],
        [_p("Frames sampled", styles["label"]), _p(data.get("frames_sampled", 0), styles["value"])],
        [_p("Pipeline status", styles["label"]), _p(verdict.get("status", "N/A"), styles["value"])],
    ]
    evidence_table = Table(evidence_rows, colWidths=[46 * mm, 112 * mm], hAlign="LEFT")
    evidence_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(evidence_table)

    # Metadata table, human-readable rather than raw JSON.
    story.append(_p("Technical Metadata", styles["h1"]))
    if metadata:
        metadata_rows = [[_p("Field", styles["table_head"]), _p("Value", styles["table_head"])]]
        for key, value in metadata.items():
            label = str(key).replace("_", " ").strip().title()
            if key == "file_size_bytes":
                value = _human_size(value)
            metadata_rows.append([
                _p(label, styles["table"]),
                _p(value, styles["table"], max_chars=1800),
            ])
        metadata_table = LongTable(metadata_rows, colWidths=[48 * mm, 110 * mm], repeatRows=1, hAlign="LEFT")
        metadata_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ]))
        story.append(metadata_table)
    else:
        story.append(_p("No technical metadata was available for this media item.", styles["body"]))

    # Methodology / reader guidance
    story.append(_p("Reader Guidance", styles["h1"]))
    guidance = [
        "Visual findings describe model-observed artifacts such as geometry, lighting, texture, warping or temporal inconsistencies.",
        "Audio/AV findings evaluate suspicious audio characteristics and audiovisual consistency when a video contains analyzable audio.",
        "Context findings assess internal consistency of the supplied media and locally extracted metadata; they are not a substitute for independent source verification.",
        "For consequential decisions, preserve the original media and corroborate this screening result with independent forensic evidence and qualified human review.",
    ]
    for item in guidance:
        story.append(_p(f"- {item}", styles["body"]))

    if verdict.get("technical_error"):
        story.append(PageBreak())
        story.append(_p("Technical Diagnostic", styles["h1"]))
        story.append(_p(
            "The analysis service returned a technical condition. This diagnostic is included for operators and should not be interpreted as media evidence.",
            styles["body"],
        ))
        story.append(_p(verdict.get("technical_error"), styles["small"], max_chars=6000))

    story.append(Spacer(1, 5 * mm))
    disclaimer = Table(
        [[_p(
            "IMPORTANT: CineTruth AI provides automated screening indicators, not a definitive authenticity determination. "
            "Results may contain false positives or false negatives and should be reviewed alongside original-source evidence.",
            styles["small"],
        )]],
        colWidths=[158 * mm],
    )
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7ED")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#FDBA74")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(disclaimer)

    def _page(canvas, doc_obj):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(16 * mm, 12.5 * mm, width - 16 * mm, 12.5 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 8.5 * mm, f"CineTruth AI | Report {session_id}")
        canvas.drawRightString(width - 16 * mm, 8.5 * mm, f"Page {doc_obj.page}")
        if doc_obj.page > 1:
            canvas.setFont("Helvetica-Bold", 7.5)
            canvas.setFillColor(NAVY)
            canvas.drawString(16 * mm, height - 10 * mm, "CineTruth AI - Media Forensic Screening Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    logger.info("PDF report successfully generated: %s", output_filename)
    return True
