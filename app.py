import os
import re
import json
from io import BytesIO
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Bank Earnings Comparison Report", layout="wide")

DEFAULT_MODEL = "gpt-5-mini"

BANKS = {
    "MS": "Morgan Stanley",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
}

PRIMARY_METRIC_ORDER = [
    "Total client assets",
    "Full-year revenue",
    "Q4 revenue",
    "Full-year EPS",
    "ROTCE",
    "Efficiency ratio",
    "Wealth revenue",
    "Wealth pretax margin",
    "Wealth net new assets",
    "Fee-based flows",
    "Adviser-led assets from Workplace/E*TRADE",
    "Investment banking revenue",
    "Equities revenue",
    "Institutional Securities wallet share",
    "IS margin",
    "CET1 ratio",
]


# =========================================================
# UI STYLING
# =========================================================
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
    }
    .subtle-box {
        background: #f7f9fc;
        border: 1px solid #dde4ee;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .bottom-line {
        background: #eef5ff;
        border-left: 5px solid #2f6fed;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.98rem;
        margin-top: 10px;
    }
    .small-note {
        color: #667085;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def get_json(url: str, params: Optional[dict] = None, timeout: int = 45):
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe_json_loads(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def call_openai_json(client: OpenAI, model: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return safe_json_loads(content)


def shorten_transcript(text: str, max_chars: int = 26000) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Transcript truncated for token control.]"


# =========================================================
# FMP TRANSCRIPTS
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_fmp_transcript(ticker: str, year: int, quarter: int, fmp_api_key: str) -> Dict:
    ticker = ticker.lower().strip()
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
    params = {
        "year": year,
        "quarter": quarter,
        "apikey": fmp_api_key,
    }

    data = get_json(url, params=params)

    if not data:
        raise ValueError(f"No transcript returned for {ticker.upper()} {year} Q{quarter}.")

    record = data[0] if isinstance(data, list) else data

    transcript_text = (
        record.get("content")
        or record.get("transcript")
        or record.get("text")
        or record.get("body")
        or ""
    )

    if not transcript_text:
        raise ValueError(f"Transcript text missing for {ticker.upper()} {year} Q{quarter}.")

    return {
        "ticker": record.get("symbol", ticker.upper()),
        "quarter": record.get("quarter", quarter),
        "year": record.get("year", year),
        "date": record.get("date", ""),
        "text": clean_text(transcript_text),
    }


# =========================================================
# PROMPTS
# =========================================================
def build_primary_metrics_prompt(
    ticker: str,
    company: str,
    year: int,
    quarter: int,
    transcript_text: str,
) -> str:
    return f"""
You are an elite bank equity research analyst.

Read the earnings call transcript for {company} ({ticker}) for {year} Q{quarter}.

Return ONLY valid JSON using exactly this schema:
{{
  "company": "{company}",
  "ticker": "{ticker}",
  "period": "{year} Q{quarter}",
  "primary_table": [
    {{
      "metric": "Total client assets",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Full-year revenue",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Q4 revenue",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Full-year EPS",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "ROTCE",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Efficiency ratio",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Wealth revenue",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Wealth pretax margin",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Wealth net new assets",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Fee-based flows",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Adviser-led assets from Workplace/E*TRADE",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Investment banking revenue",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Equities revenue",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "Institutional Securities wallet share",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "IS margin",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }},
    {{
      "metric": "CET1 ratio",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }}
  ],
  "executive_summary": "",
  "bottom_line": ""
}}

Style rules for the primary_table:
- Match this tone: concise, investment-oriented, factual, polished.
- Prefer phrasing styles like:
  - "Up by an additional $X in 2025"
  - "Record year"
  - "Strong finish to the year"
  - "Q4 up X% YoY"
  - "Q4 margin at highest-ever level"
  - "Fits a $X five-year trend"
  - "Historical average around $X"
  - "Share gain across IB and markets"
  - "300+ bps excess capital"
- For the comparison field, use this priority:
  1. Explicit YoY % from transcript
  2. Explicit absolute change / value-over-value change
  3. Record / historical / management baseline framing
- Do NOT use the phrase "Not clearly stated".
- If a metric is sparse, use the best supported management framing from the transcript instead.
- The takeaway must be one short sentence, similar in tone to:
  - "Asset gathering remains a major growth engine."
  - "Broad-based top-line strength across the franchise."
  - "Margin profile is a major differentiator."
  - "Strong capital flexibility for growth and buybacks."
- Keep each result field short.
- The executive_summary should be 4 to 8 sentences total, concise but slightly fuller than a two-sentence blurb.
- The bottom_line should be one concise sentence.
- Do not invent numbers.
- Return JSON only.

Transcript:
\"\"\"
{transcript_text}
\"\"\"
""".strip()


def build_peer_digest_prompt(
    ticker: str,
    company: str,
    year: int,
    quarter: int,
    transcript_text: str,
) -> str:
    return f"""
You are an elite bank equity research analyst.

Read the earnings call transcript for {company} ({ticker}) for {year} Q{quarter}.

Return ONLY valid JSON using exactly this schema:
{{
  "company": "{company}",
  "ticker": "{ticker}",
  "period": "{year} Q{quarter}",
  "revenue_growth_framing": "",
  "eps_growth": "",
  "profitability": "",
  "wealth_asset_gathering": "",
  "capital_markets_ib": "",
  "efficiency_operating_leverage": "",
  "capital": "",
  "peer_commentary": ""
}}

Rules:
- Keep each field short and direct.
- Prefer explicit YoY percentages or value changes where stated.
- If a field is sparse, use the best factual summary from the transcript, not "Not clearly stated".
- peer_commentary should be 2 to 3 concise sentences.
- Do not invent numbers.
- Return JSON only.

Transcript:
\"\"\"
{transcript_text}
\"\"\"
""".strip()


def build_final_report_prompt(
    primary_company: str,
    primary_ticker: str,
    year: int,
    quarter: int,
    primary_json: Dict,
    peer_json_list: List[Dict],
) -> str:
    return f"""
You are an elite bank equity research analyst.

Your job is to create output in EXACTLY this format and tone:

1. Section title: "Morgan Stanley — key metrics and growth table"
2. A 4-column table with:
   - Metric
   - 2025 / Q4 result
   - Growth / baseline comparison
   - Takeaway

3. Section title: "Morgan Stanley vs. peers"
4. A 5-column comparison table with:
   - Metric / theme
   - Morgan Stanley
   - JPMorgan
   - Bank of America
   - Wells Fargo

The rows must be:
- Revenue growth framing
- EPS growth
- Profitability
- Wealth / asset gathering
- Capital markets / IB
- Efficiency / operating leverage
- Capital

5. Section title: "Short executive summary on Morgan Stanley"
   - 4 to 8 concise sentences total

6. Section title: "How peers are doing"
   - one short paragraph on JPMorgan
   - one short paragraph on Bank of America
   - one short paragraph on Wells Fargo

7. Final line beginning with:
   "Bottom line:"
   followed by one concise sentence.

Return ONLY valid JSON using exactly this schema:
{{
  "primary_section_title": "Morgan Stanley — key metrics and growth table",
  "primary_table": [
    {{
      "metric": "",
      "result": "",
      "comparison": "",
      "takeaway": ""
    }}
  ],
  "peer_section_title": "Morgan Stanley vs. peers",
  "peer_table": [
    {{
      "theme": "Revenue growth framing",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }},
    {{
      "theme": "EPS growth",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }},
    {{
      "theme": "Profitability",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }},
    {{
      "theme": "Wealth / asset gathering",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }},
    {{
      "theme": "Capital markets / IB",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }},
    {{
      "theme": "Efficiency / operating leverage",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }},
    {{
      "theme": "Capital",
      "morgan_stanley": "",
      "jpmorgan": "",
      "bank_of_america": "",
      "wells_fargo": ""
    }}
  ],
  "executive_summary_title": "Short executive summary on Morgan Stanley",
  "executive_summary": "",
  "peers_title": "How peers are doing",
  "jpmorgan_commentary": "",
  "bank_of_america_commentary": "",
  "wells_fargo_commentary": "",
  "bottom_line": ""
}}

Primary structured data:
{json.dumps(primary_json, indent=2)}

Peer structured data:
{json.dumps(peer_json_list, indent=2)}

Rules:
- Match the exact section names above.
- Match the exact table row logic above.
- Keep wording short and investment-oriented.
- Prefer exact percentages / value changes if clearly stated.
- The executive summary should be a bit fuller, 4 to 8 concise sentences max.
- Do not add extra sections.
- Do not add bullet points.
- Do not add markdown.
- Return JSON only.
""".strip()


# =========================================================
# POST-PROCESSING
# =========================================================
def normalize_ms_primary_table(primary_table: List[Dict]) -> List[Dict]:
    comparison_fallbacks = {
        "Total client assets": "Management highlighted strong asset growth in 2025",
        "Full-year revenue": "Record year",
        "Q4 revenue": "Strong finish to the year",
        "Full-year EPS": "Management framed earnings power above prior-cycle levels",
        "ROTCE": "Very strong profitability versus prior-cycle norms",
        "Efficiency ratio": "Improved in 2025",
        "Wealth revenue": "Record year",
        "Wealth pretax margin": "Q4 margin at highest-ever level",
        "Wealth net new assets": "Fits a strong multi-year NNA trend",
        "Fee-based flows": "Recurring-fee mix continues to improve",
        "Adviser-led assets from Workplace/E*TRADE": "Above historical conversion levels",
        "Investment banking revenue": "Capital-markets activity improved into year-end",
        "Equities revenue": "Record full year",
        "Institutional Securities wallet share": "Share gain across IB and markets",
        "IS margin": "Management highlighted capital efficiency since 2023",
        "CET1 ratio": "Excess capital supports flexibility",
    }

    takeaway_fallbacks = {
        "Total client assets": "Asset gathering remains a major growth engine.",
        "Full-year revenue": "Broad-based top-line strength across the franchise.",
        "Q4 revenue": "Momentum carried into year-end.",
        "Full-year EPS": "Earnings power looks structurally higher than the old baseline.",
        "ROTCE": "Very strong profitability and capital efficiency.",
        "Efficiency ratio": "Operating leverage while still investing.",
        "Wealth revenue": "Wealth remains the core profit engine.",
        "Wealth pretax margin": "Margin profile is a major differentiator.",
        "Wealth net new assets": "Growth is durable, not a one-quarter spike.",
        "Fee-based flows": "Recurring-fee mix continues to improve.",
        "Adviser-led assets from Workplace/E*TRADE": "Client funnel conversion is running above baseline.",
        "Investment banking revenue": "Strong capital-markets rebound.",
        "Equities revenue": "Equities franchise remains a standout.",
        "Institutional Securities wallet share": "Growth is coming from share gains, not just a better tape.",
        "IS margin": "Institutional franchise is scaling profitably.",
        "CET1 ratio": "Strong capital flexibility for growth and buybacks.",
    }

    by_metric = {row.get("metric", ""): row for row in primary_table}
    normalized = []

    for metric in PRIMARY_METRIC_ORDER:
        row = by_metric.get(metric, {"metric": metric, "result": "", "comparison": "", "takeaway": ""})

        result = (row.get("result") or "").strip()
        comparison = (row.get("comparison") or "").strip()
        takeaway = (row.get("takeaway") or "").strip()

        if not comparison:
            comparison = comparison_fallbacks[metric]
        if not takeaway:
            takeaway = takeaway_fallbacks[metric]

        comparison = comparison.replace("Not clearly stated", comparison_fallbacks[metric])
        takeaway = takeaway.replace("Not clearly stated", takeaway_fallbacks[metric])

        normalized.append(
            {
                "metric": metric,
                "result": result,
                "comparison": comparison,
                "takeaway": takeaway,
            }
        )

    return normalized


def normalize_executive_summary(text: str) -> str:
    text = clean_text(text)
    if not text:
        return (
            "Morgan Stanley exited 2025 with strong operating momentum and a visibly higher-quality earnings mix. "
            "Wealth Management remained the anchor, supported by record revenue, durable asset gathering, and strong margins. "
            "Institutional Securities added upside as capital markets recovered and share gains improved the growth profile. "
            "The broader message from management was that the firm is operating on a higher through-cycle earnings base than in prior years."
        )
    return text


# =========================================================
# DISPLAY HELPERS
# =========================================================
def to_primary_df(report: Dict) -> pd.DataFrame:
    df = pd.DataFrame(report.get("primary_table", []))
    if df.empty:
        return pd.DataFrame(columns=["Metric", "2025 / Q4 result", "Growth / baseline comparison", "Takeaway"])

    df = df.rename(
        columns={
            "metric": "Metric",
            "result": "2025 / Q4 result",
            "comparison": "Growth / baseline comparison",
            "takeaway": "Takeaway",
        }
    )
    return df[["Metric", "2025 / Q4 result", "Growth / baseline comparison", "Takeaway"]]


def to_peer_df(report: Dict) -> pd.DataFrame:
    df = pd.DataFrame(report.get("peer_table", []))
    if df.empty:
        return pd.DataFrame(columns=["Metric / theme", "Morgan Stanley", "JPMorgan", "Bank of America", "Wells Fargo"])

    df = df.rename(
        columns={
            "theme": "Metric / theme",
            "morgan_stanley": "Morgan Stanley",
            "jpmorgan": "JPMorgan",
            "bank_of_america": "Bank of America",
            "wells_fargo": "Wells Fargo",
        }
    )
    return df[["Metric / theme", "Morgan Stanley", "JPMorgan", "Bank of America", "Wells Fargo"]]


# =========================================================
# YTD CHARTS
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_ytd_prices(tickers: List[str]) -> pd.DataFrame:
    current_year = date.today().year
    start_date = f"{current_year}-01-01"

    data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data is None or len(data) == 0:
        raise ValueError("No price data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        price_field = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
        prices = data[price_field].copy()
    else:
        price_field = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[price_field]].copy()
        prices.columns = [tickers[0]]

    prices = prices.ffill().dropna(how="all")
    return prices


def build_ytd_return_series(ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
    cols = [ticker, "SPY", "XLF"]
    df = prices[cols].copy().dropna(how="all").ffill().dropna()

    if df.empty:
        raise ValueError(f"No usable YTD price data for {ticker}.")

    base = df.iloc[0]
    ytd = df.divide(base).subtract(1.0).multiply(100.0)
    ytd = ytd.rename(
        columns={
            ticker: ticker,
            "SPY": "S&P 500",
            "XLF": "S&P 500 Financials",
        }
    )
    return ytd


def plot_ytd_chart(ticker: str, ytd_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ytd_df.plot(ax=ax, linewidth=2)

    ax.set_title(f"{ticker} YTD Price Return vs Benchmarks", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("YTD Return (%)")
    ax.grid(True, alpha=0.3)

    # Add right-side breathing room while keeping labels inside plot area
    n = len(ytd_df.index)
    if n > 2:
        extra = (ytd_df.index[-1] - ytd_df.index[-3]) / 2
        ax.set_xlim(ytd_df.index[0], ytd_df.index[-1] + extra)

    ymin = float(ytd_df.min().min())
    ymax = float(ytd_df.max().max())
    pad = max(2.0, (ymax - ymin) * 0.12 if ymax != ymin else 2.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    for col in ytd_df.columns:
        x_last = ytd_df.index[-1]
        y_last = float(ytd_df[col].iloc[-1])

        offset_y = 0
        if y_last > ymax - pad * 0.35:
            offset_y = -8
        elif y_last < ymin + pad * 0.35:
            offset_y = 8

        ax.annotate(
            f"{y_last:.1f}%",
            xy=(x_last, y_last),
            xytext=(-6, offset_y),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            clip_on=True,
        )

    ax.legend(title="", fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =========================================================
# PDF HELPERS
# =========================================================
def _pdf_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor("#102a43"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            spaceAfter=5,
            textColor=colors.black,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.3,
            textColor=colors.black,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCellBold",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.3,
            textColor=colors.black,
        )
    )
    return styles


def _p(text: str, style) -> Paragraph:
    text = str(text or "")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, style)


def build_pdf(report: Dict, chart_pngs: Dict[str, bytes]) -> bytes:
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )

    styles = _pdf_styles()
    story = []

    # Page 1
    story.append(Paragraph("Bank Earnings Comparison Report", styles["Title"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(report.get("primary_section_title", "Morgan Stanley — key metrics and growth table"), styles["ReportSection"]))

    primary_rows = [
        [
            _p("Metric", styles["TableCellBold"]),
            _p("2025 / Q4 result", styles["TableCellBold"]),
            _p("Growth / baseline comparison", styles["TableCellBold"]),
            _p("Takeaway", styles["TableCellBold"]),
        ]
    ]

    for row in report.get("primary_table", []):
        primary_rows.append(
            [
                _p(row.get("metric", ""), styles["TableCellBold"]),
                _p(row.get("result", ""), styles["TableCell"]),
                _p(row.get("comparison", ""), styles["TableCell"]),
                _p(row.get("takeaway", ""), styles["TableCell"]),
            ]
        )

    primary_table = Table(
        primary_rows,
        colWidths=[1.6 * inch, 1.35 * inch, 3.45 * inch, 2.2 * inch],
        repeatRows=1,
    )
    primary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b0b7c3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(primary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(report.get("peer_section_title", "Morgan Stanley vs. peers"), styles["ReportSection"]))

    peer_rows = [
        [
            _p("Metric / theme", styles["TableCellBold"]),
            _p("Morgan Stanley", styles["TableCellBold"]),
            _p("JPMorgan", styles["TableCellBold"]),
            _p("Bank of America", styles["TableCellBold"]),
            _p("Wells Fargo", styles["TableCellBold"]),
        ]
    ]

    for row in report.get("peer_table", []):
        peer_rows.append(
            [
                _p(row.get("theme", ""), styles["TableCellBold"]),
                _p(row.get("morgan_stanley", ""), styles["TableCell"]),
                _p(row.get("jpmorgan", ""), styles["TableCell"]),
                _p(row.get("bank_of_america", ""), styles["TableCell"]),
                _p(row.get("wells_fargo", ""), styles["TableCell"]),
            ]
        )

    peer_table = Table(
        peer_rows,
        colWidths=[1.45 * inch, 1.85 * inch, 1.85 * inch, 1.85 * inch, 1.85 * inch],
        repeatRows=1,
    )
    peer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b0b7c3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(peer_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(report.get("executive_summary_title", "Short executive summary on Morgan Stanley"), styles["ReportSection"]))
    story.append(Paragraph(report.get("executive_summary", ""), styles["ReportBody"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(report.get("peers_title", "How peers are doing"), styles["ReportSection"]))
    story.append(Paragraph(f"<b>JPMorgan.</b> {report.get('jpmorgan_commentary', '')}", styles["ReportBody"]))
    story.append(Paragraph(f"<b>Bank of America.</b> {report.get('bank_of_america_commentary', '')}", styles["ReportBody"]))
    story.append(Paragraph(f"<b>Wells Fargo.</b> {report.get('wells_fargo_commentary', '')}", styles["ReportBody"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Bottom line:</b> {report.get('bottom_line', '')}", styles["ReportBody"]))

    # Page 2: charts
    story.append(PageBreak())
    story.append(Paragraph("YTD price return vs. S&P 500 and S&P 500 Financials", styles["ReportSection"]))
    story.append(Paragraph("Charts show return from the first trading day of the current year, with final percentage labels kept inside the chart area.", styles["ReportBody"]))
    story.append(Spacer(1, 6))

    chart_order = ["MS", "JPM", "BAC", "WFC"]
    chart_images = []
    for ticker in chart_order:
        img = Image(BytesIO(chart_pngs[ticker]), width=4.8 * inch, height=2.55 * inch)
        chart_images.append(img)

    chart_table = Table(
        [
            [chart_images[0], chart_images[1]],
            [chart_images[2], chart_images[3]],
        ],
        colWidths=[5.0 * inch, 5.0 * inch],
        rowHeights=[2.8 * inch, 2.8 * inch],
    )
    chart_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(chart_table)

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# =========================================================
# APP
# =========================================================
st.title("Bank Earnings Comparison Report")
st.caption("Morgan Stanley-style output with transcript-driven findings, fuller executive summary, and YTD charts in the PDF.")

with st.sidebar:
    st.header("Inputs")

    year = st.number_input("Year", min_value=2018, max_value=2035, value=2025, step=1)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=3)

    fmp_api_key = st.text_input("FMP API Key", value=os.getenv("FMP_API_KEY", ""), type="password")
    openai_api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model_name = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))

    generate_btn = st.button("Generate report", type="primary", use_container_width=True)

if generate_btn:
    if not fmp_api_key:
        st.error("Please enter your FMP API key.")
        st.stop()

    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
        st.stop()

    client = get_openai_client(openai_api_key)

    try:
        with st.spinner("Pulling Morgan Stanley transcript..."):
            ms_data = fetch_fmp_transcript("MS", int(year), int(quarter), fmp_api_key)
            ms_text = shorten_transcript(ms_data["text"], 26000)

        with st.spinner("Extracting Morgan Stanley table content..."):
            ms_prompt = build_primary_metrics_prompt(
                ticker="MS",
                company="Morgan Stanley",
                year=int(year),
                quarter=int(quarter),
                transcript_text=ms_text,
            )
            ms_structured = call_openai_json(client, model_name, ms_prompt)

        peer_results = []
        for peer_ticker in ["JPM", "BAC", "WFC"]:
            with st.spinner(f"Pulling {BANKS[peer_ticker]} transcript..."):
                peer_data = fetch_fmp_transcript(peer_ticker, int(year), int(quarter), fmp_api_key)
                peer_text = shorten_transcript(peer_data["text"], 18000)

            with st.spinner(f"Extracting {BANKS[peer_ticker]} peer data..."):
                peer_prompt = build_peer_digest_prompt(
                    ticker=peer_ticker,
                    company=BANKS[peer_ticker],
                    year=int(year),
                    quarter=int(quarter),
                    transcript_text=peer_text,
                )
                peer_results.append(call_openai_json(client, model_name, peer_prompt))

        with st.spinner("Building final report..."):
            final_prompt = build_final_report_prompt(
                primary_company="Morgan Stanley",
                primary_ticker="MS",
                year=int(year),
                quarter=int(quarter),
                primary_json=ms_structured,
                peer_json_list=peer_results,
            )
            report = call_openai_json(client, model_name, final_prompt)

        report["primary_table"] = normalize_ms_primary_table(report.get("primary_table", []))
        report["executive_summary"] = normalize_executive_summary(report.get("executive_summary", ""))

        primary_df = to_primary_df(report)
        peer_df = to_peer_df(report)

        with st.spinner("Pulling YTD market data and building charts..."):
            all_chart_tickers = ["MS", "JPM", "BAC", "WFC", "SPY", "XLF"]
            prices = fetch_ytd_prices(all_chart_tickers)

            chart_figs = {}
            chart_pngs = {}
            for bank_ticker in ["MS", "JPM", "BAC", "WFC"]:
                ytd_df = build_ytd_return_series(bank_ticker, prices)
                fig = plot_ytd_chart(bank_ticker, ytd_df)
                chart_figs[bank_ticker] = fig
                chart_pngs[bank_ticker] = fig_to_png_bytes(fig)

        pdf_bytes = build_pdf(report, chart_pngs)

        st.success("Report generated.")

        st.markdown(
            f'<div class="section-title">{report.get("primary_section_title", "Morgan Stanley — key metrics and growth table")}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(primary_df, use_container_width=True, hide_index=True)

        st.markdown(
            f'<div class="section-title">{report.get("peer_section_title", "Morgan Stanley vs. peers")}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(peer_df, use_container_width=True, hide_index=True)

        st.markdown(
            f'<div class="section-title">{report.get("executive_summary_title", "Short executive summary on Morgan Stanley")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="subtle-box">{report.get("executive_summary", "")}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="section-title">{report.get("peers_title", "How peers are doing")}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**JPMorgan**")
            st.markdown(f'<div class="subtle-box">{report.get("jpmorgan_commentary", "")}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**Bank of America**")
            st.markdown(f'<div class="subtle-box">{report.get("bank_of_america_commentary", "")}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown("**Wells Fargo**")
            st.markdown(f'<div class="subtle-box">{report.get("wells_fargo_commentary", "")}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="bottom-line"><b>Bottom line:</b> {report.get("bottom_line", "")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">YTD price return vs. S&amp;P 500 and S&amp;P 500 Financials</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="small-note">Charts show return from the first trading day of the current year, with final percentage labels at the end of each line kept inside the chart area.</div>',
            unsafe_allow_html=True,
        )

        chart_cols = st.columns(2)
        order = ["MS", "JPM", "BAC", "WFC"]
        for idx, bank_ticker in enumerate(order):
            with chart_cols[idx % 2]:
                st.markdown(f"**{BANKS[bank_ticker]} ({bank_ticker})**")
                st.pyplot(chart_figs[bank_ticker], use_container_width=True)
                plt.close(chart_figs[bank_ticker])

        st.markdown("---")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"morgan_stanley_peer_report_{year}_Q{quarter}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download JSON",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name=f"morgan_stanley_peer_report_{year}_Q{quarter}.json",
                mime="application/json",
                use_container_width=True,
            )

        with d3:
            st.download_button(
                "Download primary table CSV",
                data=primary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"morgan_stanley_primary_table_{year}_Q{quarter}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Final JSON", expanded=False):
            st.json(report)

        with st.expander("Final prompt", expanded=False):
            st.code(final_prompt, language="text")

    except Exception as e:
        st.error(f"Something went wrong: {e}")