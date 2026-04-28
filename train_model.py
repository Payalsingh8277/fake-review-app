# train_model.py (upgraded)
import pandas as pd
import numpy as np
import joblib, os, re, string
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

# --- Feature Engineering ---
def extract_handcrafted_features(texts):
    """Extract linguistic signals common in fake reviews."""
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
    if uw in text.lower()),                                          # urgency phrases
            len(sentences),                                     # sentence count
            np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,    # avg sentence length
        ])
    return np.array(features)

# --- Load real dataset OR generate synthetic ---
def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        print(f"✅ Loaded real dataset: {len(df)} rows")
        print(f"   Columns found: {df.columns.tolist()}")

        # Handle 'deceptive' + 'text' column format (Ott corpus)
        if "deceptive" in df.columns and "text" in df.columns:
            df["label"]  = (df["deceptive"].str.strip().str.lower() == "deceptive").astype(int)
            df["review"] = df["text"].astype(str)

        # Handle 'label' + 'review' format (already formatted)
        elif "label" in df.columns and "review" in df.columns:
            pass

        # Handle any other text column names
        else:
            text_col  = [c for c in df.columns if "text" in c or "review" in c]
            label_col = [c for c in df.columns if "label" in c or "decept" in c or "fake" in c]
            if text_col and label_col:
                df["review"] = df[text_col[0]].astype(str)
                df["label"]  = (df[label_col[0]].str.strip().str.lower()
                                .isin(["deceptive","fake","1"])).astype(int)
            else:
                raise ValueError(f"Cannot find text/label columns. Found: {df.columns.tolist()}")

        df = df[["review", "label"]].dropna()
        print(f"   Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")
        return df.reset_index(drop=True)
    else:
        print("⚠️  No dataset found. Using synthetic data.")
        return generate_synthetic_data()

def generate_synthetic_data():
    real = [
        "Great product! Works exactly as described.",
        "Delivery was fast and packaging was excellent.",
        "I've been using this for 3 months, very durable.",
        "Solid build quality, worth every penny.",
        "Not perfect but does the job well enough.",
        "Returned it — didn't fit my needs, but quality is fine.",
        "The material feels premium. Happy with my purchase.",
        "Battery life is decent, charges quickly too.",
        "Good for the price. Minor scratches on delivery though.",
        "Used it daily for 2 weeks, still works perfectly.",
        "Honestly surprised by the quality at this price point.",
        "Setup was straightforward. Kids love it.",
        "Arrived earlier than expected. Packaging intact.",
        "Does what it says. No complaints so far.",
        "Not the best I've used, but definitely good value.",
    ] * 40

    fake = [
        "BEST PRODUCT EVER!!! BUY NOW!!! AMAZING!!!",
        "5 stars!! Perfect!! Recommend to everyone!!!",
        "Incredible! Changed my life! Order immediately!",
        "WOW WOW WOW best purchase of my entire life!!!",
        "SUPER FAST SHIPPING BEST SELLER MUST BUY!!!",
        "Amazing product!! Everyone should buy this!!!!!",
        "Perfect perfect perfect!! Zero complaints!!!",
        "LOVE LOVE LOVE THIS!!! 100/10 would recommend!!!",
        "This product is a miracle! Unbelievable quality!!!",
        "Best. Product. Ever. Period. No questions asked!!!",
        "I NEVER write reviews but this is PERFECT PERFECT PERFECT!!!",
        "BUY THIS NOW you will NOT regret it AMAZING AMAZING!!!",
        "Absolutely LOVE IT best thing I have EVER purchased!!!",
        "PERFECT GIFT!!! EVERYONE SHOULD OWN THIS!!!",
        "OUTSTANDING!! 10/10!! ORDER IMMEDIATELY!!!",
    ] * 40

    texts  = real + fake
    labels = [0] * len(real) + [1] * len(fake)
    df = pd.DataFrame({"review": texts, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Train ---
def train(csv_path=None):
    os.makedirs("model", exist_ok=True)

    df = load_data(csv_path)
    X_text = df["review"].astype(str).tolist()
    y      = df["label"].values

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                 stop_words="english", sublinear_tf=True)
    X_tfidf = vectorizer.fit_transform(X_text)

    # Handcrafted features
    X_hand  = csr_matrix(extract_handcrafted_features(X_text))

    # Combined feature matrix
    X_combined = hstack([X_tfidf, X_hand])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensemble: Logistic Regression + Random Forest
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight="balanced", n_jobs=-1)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    cv_scores = cross_val_score(ensemble, X_combined, y, cv=5, scoring="f1")

    print(f"\n✅ Test Accuracy : {accuracy_score(y_test, y_pred):.2%}")
    print(f"✅ CV F1 Score   : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Save artifacts
    joblib.dump(ensemble,   "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("💾 Saved model/model.pkl and model/vectorizer.pkl")

if __name__ == "__main__":
    train(csv_path="data/deceptive-opinion.csv")