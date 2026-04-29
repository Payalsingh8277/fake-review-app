# app.py
import re
import string

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack


st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔎",
    layout="wide",
)


st.markdown(
    """
<style>
:root {
    --bg: #f6f8fb;
    --surface: #ffffff;
    --surface-soft: #f9fbff;
    --ink: #111827;
    --muted: #64748b;
    --line: #e5eaf1;
    --navy: #0f172a;
    --blue: #2563eb;
    --cyan: #06b6d4;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --shadow: 0 18px 50px rgba(15, 23, 42, .08);
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(37, 99, 235, .10), transparent 32%),
        radial-gradient(circle at 88% 8%, rgba(6, 182, 212, .11), transparent 30%),
        var(--bg);
    color: var(--ink);
}

.block-container {
    max-width: 1220px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3, p { letter-spacing: 0; }

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, .98), rgba(30, 58, 138, .94)),
        var(--navy);
    color: white;
    padding: 34px;
    box-shadow: 0 26px 80px rgba(15, 23, 42, .22);
    margin-bottom: 18px;
}

.hero:after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px);
    background-size: 36px 36px;
    mask-image: linear-gradient(90deg, rgba(0,0,0,.9), transparent 76%);
    pointer-events: none;
}

.hero-inner {
    position: relative;
    z-index: 1;
    max-width: 780px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.14);
    color: #dbeafe;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.hero h1 {
    margin: 16px 0 10px;
    color: #ffffff;
    font-size: 46px;
    line-height: 1.05;
    font-weight: 900;
}

.hero p {
    margin: 0;
    max-width: 720px;
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.65;
}

.hero-cta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.hero-chip {
    padding: 9px 12px;
    border-radius: 999px;
    color: #e0f2fe;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.13);
    font-size: 13px;
    font-weight: 750;
}

.hero-panel {
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 8px;
    padding: 18px;
    backdrop-filter: blur(12px);
}

.hero.compact {
    padding: 38px;
    margin-bottom: 24px;
}

.hero-panel-title {
    color: #dbeafe;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-panel-row {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,.10);
}

.hero-panel-row:last-child { border-bottom: 0; }
.hero-panel-row span { color: #cbd5e1; font-size: 13px; }
.hero-panel-row strong { color: white; font-size: 18px; }

.metric-card {
    position: relative;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px;
    min-height: 118px;
    box-shadow: var(--shadow);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}

.main-shell {
    max-width: 900px;
    margin: 0 auto;
}

.result-wrap {
    margin-top: 20px;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 22px 70px rgba(15, 23, 42, .12);
    border-color: rgba(37, 99, 235, .25);
}

.metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
}

.metric-icon {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #eff6ff;
    color: var(--blue);
    font-size: 18px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 14px;
    color: var(--ink);
    font-size: 30px;
    line-height: 1;
    font-weight: 900;
}

.metric-note {
    color: var(--muted);
    font-size: 13px;
    margin-top: 7px;
}

.section-card {
    background: rgba(255,255,255,.88);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 20px;
    box-shadow: var(--shadow);
}

.section-title {
    color: var(--ink);
    font-size: 21px;
    font-weight: 900;
    margin-bottom: 4px;
}

.section-copy {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 16px;
}

.result-card {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    padding: 20px 22px;
    border: 1px solid var(--line);
    background: var(--surface);
    box-shadow: var(--shadow);
}

.result-card:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 6px;
}

.result-card.fake { border-color: rgba(239, 68, 68, .28); background: #fff7f7; }
.result-card.fake:before { background: var(--red); }
.result-card.real { border-color: rgba(16, 185, 129, .28); background: #f5fffa; }
.result-card.real:before { background: var(--green); }
.result-card.uncertain { border-color: rgba(245, 158, 11, .32); background: #fffaf0; }
.result-card.uncertain:before { background: var(--amber); }

.result-label {
    color: var(--ink);
    font-size: 25px;
    font-weight: 900;
}

.result-subtitle {
    margin-top: 4px;
    color: var(--muted);
    font-size: 14px;
}

.risk-band {
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px 16px;
    margin: 12px 0 18px;
}

.risk-title {
    color: var(--ink);
    font-size: 15px;
    font-weight: 900;
}

.risk-copy {
    color: var(--muted);
    font-size: 13px;
    margin-top: 3px;
}

.plain-result {
    border-radius: 8px;
    padding: 24px;
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    background: var(--surface);
}

.plain-result.fake {
    background: linear-gradient(135deg, #fff7f7, #ffffff);
    border-color: rgba(239, 68, 68, .30);
}

.plain-result.real {
    background: linear-gradient(135deg, #f5fffa, #ffffff);
    border-color: rgba(16, 185, 129, .30);
}

.plain-result.uncertain {
    background: linear-gradient(135deg, #fffaf0, #ffffff);
    border-color: rgba(245, 158, 11, .34);
}

.result-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 14px;
}

.result-status.fake { background: #fee2e2; color: #991b1b; }
.result-status.real { background: #dcfce7; color: #166534; }
.result-status.uncertain { background: #fef3c7; color: #92400e; }

.plain-result h2 {
    margin: 0;
    font-size: 30px;
    color: var(--ink);
    line-height: 1.15;
}

.plain-result p {
    color: var(--muted);
    font-size: 15px;
    margin: 8px 0 0;
}

.confidence-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-top: 20px;
    color: var(--ink);
    font-weight: 900;
}

.confidence-track {
    height: 12px;
    background: #e8eef6;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
}

.confidence-fill.fake { background: linear-gradient(90deg, #fb7185, var(--red)); }
.confidence-fill.real { background: linear-gradient(90deg, #34d399, var(--green)); }
.confidence-fill.uncertain { background: linear-gradient(90deg, #fbbf24, var(--amber)); }

.explain-card {
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
}

.explain-title {
    color: var(--ink);
    font-size: 15px;
    font-weight: 900;
    margin-bottom: 10px;
}

.explain-item {
    display: flex;
    gap: 10px;
    color: #334155;
    font-size: 14px;
    padding: 8px 0;
    border-bottom: 1px solid #e8eef6;
}

.explain-item:last-child { border-bottom: 0; }

.simple-list {
    display: grid;
    gap: 10px;
}

.simple-list-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px 14px;
}

.simple-list-title {
    color: var(--ink);
    font-weight: 850;
}

.simple-list-copy {
    color: var(--muted);
    font-size: 13px;
}

.score-badge {
    white-space: nowrap;
    border-radius: 999px;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 900;
}

.score-badge.fake { background: #fee2e2; color: #991b1b; }
.score-badge.real { background: #dcfce7; color: #166534; }
.score-badge.uncertain { background: #fef3c7; color: #92400e; }

div[data-testid="stTabs"] {
    background: rgba(255,255,255,.64);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 8px 0;
}

div[data-testid="stTabs"] button {
    border-radius: 8px 8px 0 0;
    color: var(--muted);
    font-weight: 850;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    background: #ffffff;
    color: var(--blue);
    border-bottom: 3px solid var(--blue);
}

.stTextArea textarea {
    border-radius: 8px;
    border: 1px solid #d8e0ea;
    background: #ffffff;
    color: var(--ink);
    font-size: 15px;
}

.stButton button, .stDownloadButton button {
    border-radius: 8px;
    border: 1px solid rgba(37, 99, 235, .20);
    font-weight: 900;
    transition: transform .16s ease, box-shadow .16s ease;
}

.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 34px rgba(37, 99, 235, .18);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
}

@media (max-width: 820px) {
    .hero h1 { font-size: 36px; }
    .hero { padding: 24px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def extract_handcrafted_features(texts):
    features = []
    for text in texts:
        words = text.split()
        sentences = re.split(r"[.!?]", text)
        features.append(
            [
                len(text),
                len(words),
                text.count("!"),
                text.count("?"),
                sum(1 for c in text if c.isupper()) / max(len(text), 1),
                len([w for w in words if w.isupper()]) / max(len(words), 1),
                sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
                len(set(words)) / max(len(words), 1),
                sum(text.lower().count(w) for w in ["best", "perfect", "amazing", "love", "excellent"]),
                sum(
                    1
                    for phrase in [
                        "buy now",
                        "act now",
                        "hurry",
                        "dont miss",
                        "grab it",
                        "order immediately",
                        "buy immediately",
                        "purchase now",
                    ]
                    if phrase in text.lower()
                ),
                len(sentences),
                np.mean([len(s.split()) for s in sentences if s.strip()])
                if any(s.strip() for s in sentences)
                else 0.0,
            ]
        )
    return np.array(features)


@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


def has_high_risk_fake_signal(text):
    lower_text = text.lower()
    words = text.split()
    urgency_phrases = [
        "buy now",
        "act now",
        "hurry",
        "dont miss",
        "grab it",
        "order immediately",
        "buy immediately",
        "purchase now",
    ]
    superlatives = ["best", "perfect", "amazing", "excellent", "outstanding", "incredible"]
    urgency_count = sum(1 for phrase in urgency_phrases if phrase in lower_text)
    superlative_count = sum(lower_text.count(word) for word in superlatives)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    all_caps_words = sum(1 for word in words if len(word) > 2 and word.isupper())

    return (
        urgency_count > 0
        or (text.count("!") >= 3 and (uppercase_ratio > 0.18 or superlative_count >= 2))
        or all_caps_words >= 4
    )


def predict(texts):
    X_tfidf = vectorizer.transform(texts)
    X_hand = csr_matrix(extract_handcrafted_features(texts))
    X = hstack([X_tfidf, X_hand])
    preds = model.predict(X)
    probas = model.predict_proba(X)

    for i, text in enumerate(texts):
        if has_high_risk_fake_signal(text):
            probas[i][1] = max(probas[i][1], 0.85)
            probas[i][0] = 1 - probas[i][1]
            preds[i] = 1

    return preds, probas


def clean_amazon_paste(raw_text):
    lines = raw_text.split("\n")
    reviews = []
    current = []
    skip_patterns = [
        r"^\d+\.\d+ out of \d+ stars",
        r"^Reviewed in .+ on ",
        r"^Size:.*Colour:",
        r"^Verified Purchase",
        r"^\d+ people found this helpful",
        r"^One person found this helpful",
        r"^Helpful$",
        r"^Report$",
        r"^[A-Z]{1,3}$",
        r"^\[Report\]",
        r"^\[.*\]\(https?://",
        r"^https?://",
        r"^\*\s+\[",
        r"^\* $",
        r"^amazon\.in",
        r"^\[.*\]\(.*amazon",
    ]

    for line in lines:
        line = line.strip()
        line = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", line)
        line = line.strip("* ").strip()

        if not line:
            if current:
                reviews.append(" ".join(current))
                current = []
            continue

        skip = any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns)
        if not skip and len(line) > 15:
            current.append(line)

    if current:
        reviews.append(" ".join(current))

    return [review for review in reviews if len(review) > 20]


def prediction_label(pred, p_fake, p_real, threshold=0.65):
    if pred == 1 and p_fake >= threshold:
        return "🚨 Fake"
    if pred == 0 and p_real >= threshold:
        return "✅ Real"
    return "⚠️ Uncertain"


def confidence_text(p_fake, p_real):
    confidence = max(p_fake, p_real) * 100
    if confidence >= 85:
        return "High confidence"
    if confidence >= 65:
        return "Moderate confidence"
    return "Needs review"


def metric_card(icon, label, value, note):
    return f"""
    <div class="metric-card">
        <div class="metric-top">
            <div>
                <div class="metric-label">{label}</div>
            </div>
            <div class="metric-icon">{icon}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def result_card(label, p_fake, p_real):
    if label == "🚨 Fake":
        class_name = "fake"
        title = "Fake Review Detected"
        subtitle = f"Fake score {p_fake * 100:.1f}% · {confidence_text(p_fake, p_real)}"
    elif label == "✅ Real":
        class_name = "real"
        title = "Genuine Review"
        subtitle = f"Real score {p_real * 100:.1f}% · {confidence_text(p_fake, p_real)}"
    else:
        class_name = "uncertain"
        title = "⚠️ Uncertain Review"
        subtitle = f"Closest score {max(p_fake, p_real) * 100:.1f}% · manual review recommended"

    return f"""
    <div class="result-card {class_name}">
        <div class="result-label">{title}</div>
        <div class="result-subtitle">{subtitle}</div>
    </div>
    """


def user_result(label, p_fake, p_real):
    if label == "🚨 Fake":
        return "fake", "Likely Fake Review", "This review contains patterns that may appear suspicious or promotional.", p_fake
    if label == "✅ Real":
        return "real", "Genuine Review", "This review looks more natural and balanced.", p_real
    return "uncertain", "Uncertain Review", "The app is not confident enough to make a strong decision.", max(p_fake, p_real)


def simple_explanations(text, label):
    lower_text = text.lower()
    words = text.split()
    reasons = []

    if any(phrase in lower_text for phrase in ["buy now", "act now", "hurry", "grab it", "order immediately", "purchase now"]):
        reasons.append("Contains pushy words that encourage quick buying.")
    if text.count("!") > 2:
        reasons.append("Uses many exclamation marks, which can make a review feel promotional.")
    if sum(c.isupper() for c in text) / max(len(text), 1) > 0.1:
        reasons.append("Uses many capital letters, which can look unnatural.")
    if any(word in lower_text for word in ["perfect", "amazing", "best", "excellent", "outstanding"]):
        reasons.append("Contains very strong praise words.")
    if len(words) < 25:
        reasons.append("Provides limited detail.")
    if len(set(words)) / max(len(words), 1) > 0.7 and len(words) >= 25:
        reasons.append("Uses varied wording, which is usually a positive sign.")

    if not reasons:
        if label == "✅ Real":
            reasons.append("The wording looks calm and natural.")
        elif label == "🚨 Fake":
            reasons.append("The overall wording pattern looks suspicious.")
        else:
            reasons.append("There are not enough strong signs either way.")

    return reasons[:4]


def explanation_card(text, label):
    reasons = simple_explanations(text, label)
    items = "".join(f'<div class="explain-item"><span>•</span><span>{reason}</span></div>' for reason in reasons)
    title = "Why this result?"
    return f"""
    <div class="explain-card">
        <div class="explain-title">{title}</div>
        {items}
    </div>
    """


def consumer_result_card(label, p_fake, p_real, text):
    class_name, title, subtitle, confidence = user_result(label, p_fake, p_real)
    return f"""
    <div class="plain-result {class_name}">
        <div class="result-status {class_name}">{title}</div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
        <div class="confidence-row">
            <span>Confidence</span>
            <span>{confidence * 100:.1f}%</span>
        </div>
        <div class="confidence-track">
            <div class="confidence-fill {class_name}" style="width:{confidence * 100:.1f}%"></div>
        </div>
        {explanation_card(text, label)}
    </div>
    """


def feature_signal_explanations(text, label):
    lower_text = text.lower()
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    promotional_words = ["perfect", "amazing", "best", "excellent", "outstanding", "incredible", "love"]
    urgency_phrases = ["buy now", "act now", "hurry", "grab it", "order immediately", "purchase now"]
    reasons = []

    if label == "🚨 Fake":
        if any(word in lower_text for word in promotional_words) or text.count("!") > 2:
            reasons.append("Too many promotional or exaggerated words.")
        if unique_ratio < 0.55:
            reasons.append("Repetitive phrases detected.")
        if len(words) < 25:
            reasons.append("Lacks personal experience or details.")
        if any(phrase in lower_text for phrase in urgency_phrases):
            reasons.append("Uses pushy buying language.")
        if not reasons:
            reasons.append("Looks generic and template-like.")
    elif label == "✅ Real":
        reasons.append("Natural writing style.")
        if len(words) >= 25:
            reasons.append("Includes enough detail to understand the experience.")
        if text.count("!") <= 1:
            reasons.append("Balanced tone, not overly positive.")
        if unique_ratio > 0.65:
            reasons.append("Uses varied vocabulary.")
    else:
        reasons.append("The review has mixed signals.")
        reasons.append("Manual checking is recommended.")
        if len(words) < 25:
            reasons.append("There is limited detail to judge from.")

    return reasons[:4]


def render_feature_signals(text, label):
    title = (
        "Why this review looks suspicious"
        if label == "🚨 Fake"
        else "Why this review looks genuine"
        if label == "✅ Real"
        else "Why this review needs a second look"
    )
    st.markdown(f"**{title}**")
    for reason in feature_signal_explanations(text, label):
        st.markdown(f"- {reason}")


def why_flags(review_input):
    flags = []
    words = review_input.lower().split()

    if review_input.count("!") > 2:
        flags.append(f"High exclamation count ({review_input.count('!')})")
    if sum(c.isupper() for c in review_input) / max(len(review_input), 1) > 0.1:
        flags.append("High uppercase ratio, which can indicate emphasis or shouting.")
    if len(set(words)) / max(len(words), 1) < 0.5:
        flags.append("Low vocabulary diversity, which can indicate repetitive wording.")
    if any(w in words for w in ["perfect", "amazing", "incredible", "best", "outstanding"]):
        flags.append("Contains strong promotional words.")
    if len(words) < 25:
        flags.append("Very short review, so there is limited detail to judge.")
    if any(
        phrase in review_input.lower()
        for phrase in ["buy now", "act now", "hurry", "grab it", "order immediately", "purchase now"]
    ):
        flags.append("Contains urgency language.")
    if review_input.count("!") == 0 and len(words) > 50:
        flags.append("Calm tone with no exclamation marks.")
    if len(set(words)) / max(len(words), 1) > 0.7:
        flags.append("High vocabulary variety.")
    if not flags:
        flags.append("No strong fake or real signal was detected.")

    return flags


st.markdown(
    """
    <div class="hero compact">
        <div class="hero-inner">
            <div>
                <div class="eyebrow">🔎 Review Trust Check</div>
                <h1>Fake Review Detector</h1>
                <p>
                    Check whether a product review looks genuine or suspicious in seconds.
                </p>
                <div class="hero-cta-row">
                    <span class="hero-chip">Fast analysis</span>
                    <span class="hero-chip">Reliable results</span>
                    <span class="hero-chip">Bulk checking</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab1, tab2 = st.tabs(["Check One Review", "Check Many Reviews"])


with tab1:
    left, right = st.columns([1.45, 1], gap="large")
    analyzed = False
    analyzed_label = None

    with left:
        with st.container(border=True):
            st.markdown("### Check a Review")
            review_input = st.text_area(
                "Product review",
                height=180,
                placeholder="Paste a product review here...",
            )

            analyze_clicked = st.button("Analyze Review", type="primary", use_container_width=True)

        if analyze_clicked:
            if not review_input.strip():
                st.warning("Please enter a review first.")
            else:
                preds, probas = predict([review_input])
                pred = preds[0]
                p_fake = probas[0][1]
                p_real = probas[0][0]
                label = prediction_label(pred, p_fake, p_real)
                analyzed = True
                analyzed_label = label

                st.write("")
                st.markdown(result_card(label, p_fake, p_real), unsafe_allow_html=True)

                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.markdown(metric_card("🚨", "Fake Score", f"{p_fake * 100:.1f}%", "Suspicion probability"), unsafe_allow_html=True)
                c2.markdown(metric_card("✅", "Real Score", f"{p_real * 100:.1f}%", "Genuine probability"), unsafe_allow_html=True)
                c3.markdown(
                    metric_card("📊", "Confidence", f"{max(p_fake, p_real) * 100:.1f}%", confidence_text(p_fake, p_real)),
                    unsafe_allow_html=True,
                )
                st.progress(float(p_fake), text="Fake review risk")

                with st.expander("Why did the model predict this?"):
                    for flag in why_flags(review_input):
                        st.markdown(f"- {flag}")

    with right:
        with st.container(border=True):
            st.markdown("### Feature Signals")
            if review_input.strip() and analyzed:
                render_feature_signals(review_input, analyzed_label)
            else:
                st.info("Feature indicators will appear after your review.")


with tab2:
    with st.container(border=True):
        st.markdown("### Check Multiple Reviews")
        st.caption("Paste one review per line. Results stay simple and easy to scan.")

        if st.button("Load Sample Reviews"):
            st.session_state["multi_input"] = """Great product, works exactly as described. Very happy with purchase.
AMAZING!! BEST PRODUCT EVER!! BUY NOW!! CHANGED MY LIFE!!
Decent quality but shipping was delayed by 3 days.
PERFECT PERFECT PERFECT!! Everyone must buy this immediately!!
Battery life is okay, not great but gets the job done.
WOW WOW WOW!! Incredible quality!! Order before it sells out!!
Comfortable and lightweight. Good value for the price.
I LOVE THIS SO MUCH!! Best decision of my entire life!!"""

        bulk_text = st.text_area(
            "Reviews",
            height=230,
            value=st.session_state.get("multi_input", ""),
            placeholder="Review 1...\nReview 2...\nReview 3...",
        )

        analyze_all_clicked = st.button("Analyze All Reviews", type="primary", use_container_width=True)

    if analyze_all_clicked:
        reviews = clean_amazon_paste(bulk_text)

        if not reviews:
            st.warning("No reviews found. Paste one review per line or Amazon review text.")
        else:
            with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                preds, probas = predict(reviews)

            n_fake = int(sum(preds))
            n_real = len(preds) - n_fake
            fake_pct = n_fake / len(preds) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(metric_card("📋", "Total Reviews", len(preds), "Reviews analyzed"), unsafe_allow_html=True)
            m2.markdown(metric_card("🚨", "Fake", n_fake, "Flagged reviews"), unsafe_allow_html=True)
            m3.markdown(metric_card("✅", "Real", n_real, "Genuine reviews"), unsafe_allow_html=True)
            m4.markdown(metric_card("📌", "Fake Rate", f"{fake_pct:.1f}%", "Overall risk"), unsafe_allow_html=True)

            st.progress(
                fake_pct / 100,
                text=f"{'High' if fake_pct > 50 else 'Moderate' if fake_pct > 25 else 'Low'} fake review risk",
            )

            predictions = [
                prediction_label(p, p_fake, p_real)
                for p, p_fake, p_real in zip(preds, probas[:, 1], probas[:, 0])
            ]
            results_df = pd.DataFrame(
                {
                    "Review": reviews,
                    "Prediction": predictions,
                    "Fake Score %": (probas[:, 1] * 100).round(1),
                    "Real Score %": (probas[:, 0] * 100).round(1),
                }
            )

            filter_opt = st.radio("Show", ["All", "🚨 Fake only", "✅ Real only"], horizontal=True)
            filtered_df = results_df
            if filter_opt == "🚨 Fake only":
                filtered_df = results_df[results_df["Prediction"] == "🚨 Fake"]
            elif filter_opt == "✅ Real only":
                filtered_df = results_df[results_df["Prediction"] == "✅ Real"]

            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "Review": st.column_config.TextColumn(width="large"),
                    "Fake Score %": st.column_config.NumberColumn("Fake Score %", format="%.1f%%"),
                    "Real Score %": st.column_config.NumberColumn("Real Score %", format="%.1f%%"),
                },
            )

            csv_out = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                csv_out,
                "multi_results.csv",
                "text/csv",
                use_container_width=True,
            )
