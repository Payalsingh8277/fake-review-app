# app.py (fully upgraded)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re, string
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="wide")
g
# --- CSS ---
st.markdown("""
<style>
.result-fake { background:#ff4b4b18; border:2px solid #ff4b4b;
               color:#ff4b4b; padding:16px; border-radius:10px;
               text-align:center; font-size:20px; font-weight:700; }
.result-real { background:#00c85318; border:2px solid #00c853;
               color:#00c853; padding:16px; border-radius:10px;
               text-align:center; font-size:20px; font-weight:700; }
.feature-box { background:#1e1e2e; padding:14px; border-radius:8px;
               font-size:13px; color:#cdd6f4; }
</style>
""", unsafe_allow_html=True)

# --- Feature extractor (must match train_model.py) ---
def extract_handcrafted_features(texts):
    features = []
    for text in texts:
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        features.append([
            len(text),                                          # review length
            len(words),                                         # word count
            text.count('!'),                                    # exclamation marks
            text.count('?'),                                    # question marks
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # uppercase ratio
            len([w for w in words if w.isupper()]) / max(len(words), 1),  # all-caps word ratio
            text.count(string.punctuation) / max(len(text), 1), # punctuation density
            len(set(words)) / max(len(words), 1),               # vocabulary diversity
            sum(text.count(w) for w in ['best', 'perfect', 'amazing', 'love', 'excellent']),  # superlative count
            sum(1 for uw in ["buy now", "act now", "hurry", "dont miss", "grab it",
                 "order immediately", "buy immediately", "purchase now"]
                if uw in text.lower()),                         # urgency phrases
            len(sentences),                                     # sentence count
            np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,  # avg sentence length
        ])
    return np.array(features)

@st.cache_resource
def load_model():
    model      = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()   # ← moved up, before predict()

def predict(texts):                # ← removed model/vectorizer arguments
    X_tfidf = vectorizer.transform(texts)
    X_hand  = csr_matrix(extract_handcrafted_features(texts))
    X       = hstack([X_tfidf, X_hand])
    preds   = model.predict(X)
    probas  = model.predict_proba(X)
    return preds, probas

# ── Header ──────────────────────────────────────────────
st.title("🔍 Fake Review Detector")
st.caption("Powered by TF-IDF + Handcrafted Features + Ensemble Model")
st.divider()

# --- at the top of app.py, add this import ---
from scraper import scrape_reviews

# --- replace your tabs line with ---
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📂 Bulk CSV Upload", "🌐 Scrape from URL"])

# ══════════════════════════════════════════════
# TAB 1 — Single Review
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        review_input = st.text_area("Enter Review Text", height=160,
                                    placeholder="Paste a product review here…")
        if st.button("🔎 Analyse Review", type="primary", use_container_width=True):
            if not review_input.strip():
                st.warning("Please enter a review.")
            else:
                preds, probas = predict([review_input])
                pred   = preds[0]
                p_fake = probas[0][1]
                p_real = probas[0][0]

                # ── Confidence threshold ──────────────────
                if pred == 1 and p_fake >= 0.65:
                    st.markdown('<div class="result-fake">🚨 FAKE Review</div>',
                                unsafe_allow_html=True)
                elif pred == 0 and p_real >= 0.65:
                    st.markdown('<div class="result-real">✅ Genuine Review</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('''<div style="background:#f5a62318; border:2px solid #f5a623;
                                color:#f5a623; padding:18px; border-radius:10px;
                                text-align:center; font-size:22px; font-weight:700;">
                                🤔 Uncertain — Could be either</div>''',
                                unsafe_allow_html=True)
                    st.info("⚠️ Confidence below 65%. The model is not sure about this review. "
                            "Download the real Kaggle dataset and retrain for better accuracy.")

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Fake probability", f"{p_fake*100:.1f}%")
                c2.metric("Real probability", f"{p_real*100:.1f}%")
                c3.metric("Confidence",       f"{max(p_fake, p_real)*100:.1f}%")
                st.progress(float(p_fake), text="Fake score")
                
                # ── Why explainer ─────────────────────────
                with st.expander("🧠 Why did the model predict this?"):
                    flags = []
                    words = review_input.lower().split()

                    if review_input.count("!") > 2:
                        flags.append(f"❗ High exclamation count ({review_input.count('!')})")
                    if sum(c.isupper() for c in review_input) / max(len(review_input), 1) > 0.1:
                        flags.append("🔠 High uppercase ratio — possible shouting/emphasis")
                    if len(set(words)) / max(len(words), 1) < 0.5:
                        flags.append("📚 Low vocabulary diversity — repetitive language")
                    if any(w in words for w in ["perfect","amazing","incredible","best","outstanding"]):
                        flags.append("🌟 Contains superlative words (perfect, amazing, best…)")
                    if len(words) < 25:
                        flags.append("📏 Very short review — under 25 words")
                    if any(phrase in review_input.lower() for phrase in
                           ["buy now","act now","hurry","dont miss",
                            "grab it","order immediately","purchase now"]):
                        flags.append("⚡ Contains urgency phrases (buy now, hurry…)")
                    if review_input.count("!") == 0 and len(words) > 50:
                        flags.append("✅ Calm tone with no exclamation marks")
                    if len(set(words)) / max(len(words), 1) > 0.7:
                        flags.append("✅ High vocabulary diversity — detailed language")
                    if not flags:
                        flags.append("✅ No strong fake or real signals detected")

                    for f in flags:
                        st.markdown(f"- {f}")

                    st.caption(
                        "ℹ️ Model uses TF-IDF bigrams + 12 linguistic features. "
                        "Accuracy improves with the real Kaggle dataset."
                    )


    with col_right:
        st.markdown("#### 🧪 Feature Signals")
        if review_input.strip():
            words = review_input.split()
            st.markdown(f"""<div class="feature-box">
            📏 <b>Length</b>: {len(review_input)} chars<br>
            🔤 <b>Words</b>: {len(words)}<br>
            ❗ <b>Exclamations</b>: {review_input.count('!')}<br>
            🔠 <b>Uppercase ratio</b>: {sum(c.isupper() for c in review_input)/max(len(review_input),1)*100:.1f}%<br>
            📚 <b>Vocab diversity</b>: {len(set(words))/max(len(words),1)*100:.1f}%<br>
            ⚡ <b>Urgency words</b>: {sum(review_input.lower().count(w) for w in ["buy now", "act now", "hurry", "dont miss", "grab it",
                 "order immediately", "buy immediately", "purchase now"])}<br>
            🌟 <b>Superlatives</b>: {sum(review_input.lower().count(w) for w in ['best','perfect','amazing','love','excellent'])}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Feature signals will appear after you enter a review.")

# ══════════════════════════════════════════════
# TAB 2 — Bulk CSV Upload
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    Upload a CSV with a **`review`** column.  
    Optionally include a **`label`** column (0=real, 1=fake) to see accuracy.
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "review" not in df.columns:
            st.error("❌ CSV must have a 'review' column.")
        else:
            with st.spinner("Analysing reviews…"):
                texts         = df["review"].astype(str).tolist()
                preds, probas = predict(texts)

                df["Prediction"]    = ["🚨 Fake" if p == 1 else "✅ Real" for p in preds]
                df["Fake Score %"]  = (probas[:, 1] * 100).round(1)
                df["Real Score %"]  = (probas[:, 0] * 100).round(1)

            # Summary metrics
            n_fake = int(sum(preds))
            n_real = len(preds) - n_fake
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Reviews",  len(preds))
            m2.metric("🚨 Fake Detected", n_fake)
            m3.metric("✅ Real Reviews", n_real)

            # Accuracy if ground truth available
            if "label" in df.columns:
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(df["label"], preds)
                f1  = f1_score(df["label"], preds)
                st.success(f"Ground truth found → Accuracy: **{acc:.2%}** | F1: **{f1:.2%}**")

            # Show table
            st.divider()
            st.dataframe(
                df[["review", "Prediction", "Fake Score %", "Real Score %"]],
                use_container_width=True,
                column_config={
                    "Fake Score %": st.column_config.ProgressColumn(
                        "Fake Score %", min_value=0, max_value=100),
                }
            )

            # Download results
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "results.csv", "text/csv", use_container_width=True)
            
            # ══════════════════════════════════════════════
# TAB 3 — Scrape Reviews from URL
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🌐 Scrape & Analyse Reviews from a URL")
    st.caption("Supports Amazon, Flipkart, Yelp, TripAdvisor, and most review sites.")

    url_input = st.text_input(
        "Paste product/review page URL",
        placeholder="https://www.amazon.in/dp/B0XXXXXXXX"
    )
    st.caption("""
    💡 **Tips for best results:**
    - **Amazon**: Use product page URL → `amazon.in/dp/PRODUCTID` (not the 'see all reviews' page)
    - **Flipkart**: Use the reviews tab URL
    - **TripAdvisor**: Paste hotel page URL directly
    - **Avoid**: 'See all reviews' pages — these require login and are blocked
    """)
    max_rev    = st.number_input("Max reviews", 5, 100, 20, 5)
    scrape_btn = st.button("🕷️ Scrape & Detect", type="primary",
                           use_container_width=True)

    if scrape_btn:
        if not url_input.strip():
            st.warning("⚠️ Please enter a URL first.")
        else:
            # ── Scrape ──
            # Auto-fix Amazon URL and show user
            display_url = url_input.strip()
            if "amazon" in display_url and "/product-reviews/" in display_url:
                product_id  = display_url.split("/product-reviews/")[1].split("/")[0].split("?")[0]
                display_url = f"https://www.amazon.in/dp/{product_id}"
                st.info(f"🔄 Auto-converted to product page URL: `{display_url}`")

            with st.spinner("🔍 Fetching reviews from the page…"):
                try:
                    reviews = scrape_reviews(display_url)
                    reviews = reviews[:max_rev]
                    
                except Exception as e:
                    err = str(e)
                    st.error(f"❌ Scraping failed: {err}")

                    if "403" in err or "Forbidden" in err:
                        st.warning("⚠️ This site blocks scrapers (403 Forbidden). Try these instead:")
                    elif "No reviews found" in err:
                        st.warning("⚠️ Page loaded but no reviews detected. The site structure may have changed.")
                    elif "CAPTCHA" in err or "robot" in err.lower():
                        st.warning("⚠️ Site is showing a CAPTCHA. Cannot scrape automatically.")

                    st.info("""
                    **URLs that work well:**
                    - https://www.tripadvisor.in (hotel reviews)
                    - https://books.toscrape.com (test site)
                    - https://play.google.com/store/apps/ (app reviews)
                    """)
                    st.stop()

            st.success(f"✅ Found **{len(reviews)}** reviews. Running detection…")

            # ── Predict ──
            with st.spinner("🤖 Analysing with ML model…"):
                preds, probas = predict(reviews)

            # ── Summary metrics ──
            n_fake = int(sum(preds))
            n_real = len(preds) - n_fake
            fake_pct = n_fake / len(preds) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📋 Total Reviews",   len(preds))
            m2.metric("🚨 Fake Detected",   n_fake)
            m3.metric("✅ Real Reviews",    n_real)
            m4.metric("⚠️ Fake Rate",       f"{fake_pct:.1f}%")

            # Colour-coded fake rate bar
            st.progress(fake_pct / 100,
                        text=f"{'🔴 High' if fake_pct > 50 else '🟡 Moderate' if fake_pct > 25 else '🟢 Low'} fake rate")

            st.divider()

            # ── Results table ──
            results_df = pd.DataFrame({
                "Review":        reviews,
                "Prediction":    ["🚨 Fake" if p == 1 else "✅ Real" for p in preds],
                "Fake Score %":  (probas[:, 1] * 100).round(1),
                "Real Score %":  (probas[:, 0] * 100).round(1),
            })

            # Filter toggle
            filter_opt = st.radio("Show", ["All", "🚨 Fake only", "✅ Real only"],
                                  horizontal=True)
            if filter_opt == "🚨 Fake only":
                results_df = results_df[results_df["Prediction"] == "🚨 Fake"]
            elif filter_opt == "✅ Real only":
                results_df = results_df[results_df["Prediction"] == "✅ Real"]

            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "Review":       st.column_config.TextColumn(width="large"),
                    "Fake Score %": st.column_config.ProgressColumn(
                                        "Fake Score %", min_value=0, max_value=100),
                }
            )

            # ── Download ──
            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "scraped_results.csv", "text/csv",
                               use_container_width=True)

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("📖 How it works")
    st.markdown("""
    **Features used:**
    - TF-IDF (1–2 grams, 8000 features)
    - Review length & word count
    - Exclamation mark count
    - Uppercase ratio
    - Vocabulary diversity
    - Urgency & superlative word count

    **Model:** Soft-voting Ensemble  
    (Logistic Regression + Random Forest)

    **Fake signals to watch:**
    - Excessive `!!!`
    - ALL CAPS words
    - Words like *BUY NOW*, *PERFECT*, *AMAZING*
    - Low vocabulary diversity
    - Very short or very generic text
    """)
    st.divider()
    st.caption("Retrain anytime: `python train_model.py`")