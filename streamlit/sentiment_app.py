"""
sentiment_app.py — Streamlit Dashboard for Amazon Review Sentiment Analysis
Matches exactly: Amazon_reviews_for_sentiment_analysis.ipynb

Pipeline:
- Dataset     : amazon_reviews.csv (4,914 reviews, 90.5% positive / 9.5% negative)
- Target      : overall ≥ 4 → positive (1), else negative (0)
- Preprocessing: lowercase → regex clean → tokenize → stopword removal → lemmatization
- Features    : TF-IDF (max_features=5000, ngram_range=(1,2))
- Model       : LogisticRegression(class_weight='balanced', C=10, max_iter=1000)
- Key result  : Negative recall improved from 16% → 72% after class_weight fix

How to run:
    pip install streamlit scikit-learn nltk pandas matplotlib seaborn
    streamlit run sentiment_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

# ── Download NLTK resources ───────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(pkg, quiet=True)

download_nltk()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
    .positive { color: #2ecc71; font-weight: bold; font-size: 1.5rem; }
    .negative { color: #e74c3c; font-weight: bold; font-size: 1.5rem; }
    .metric-box {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #3498db; margin: 4px 0;
    }
    .stProgress > div > div { background-color: #3498db; }
</style>
""", unsafe_allow_html=True)


# ── Preprocessing — exact match from notebook ─────────────────────────────────
@st.cache_resource
def get_resources():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = get_resources()

def preprocess(text: str) -> str:
    """
    Exact pipeline from notebook:
    1. Lowercase
    2. Remove non-alphabetic chars (regex)
    3. Tokenize (split)
    4. Remove stopwords
    5. Lemmatize
    6. Join back
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


# ── Generate training data (mirrors dataset structure) ───────────────────────
def generate_data(uploaded_file):
    import io
    df = pd.read_csv(uploaded_file)

    # Exact target creation — notebook Cell 14
    df['overall']   = df['overall'].astype(int)
    df['sentiment'] = df['overall'].map({1:0, 2:0, 3:0, 4:1, 5:1})

    # Keep only required columns, drop nulls — notebook Cell 19-22
    df = df[['reviewText', 'sentiment']].dropna().reset_index(drop=True)

    # Preprocess — exact pipeline from notebook
    df['final_text'] = df['reviewText'].apply(preprocess)
    return df


# ── Train Model — exact params from notebook ─────────────────────────────────
def train_model(uploaded_file):
    df = generate_data(uploaded_file)

    X = df['final_text']
    y = df['sentiment']

    # TF-IDF: max_features=5000, ngram_range=(1,2) — exact from notebook
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline (no class_weight) — notebook Cell 56
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    # Balanced model — notebook Cell 62 (the FIX)
    balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
    balanced.fit(X_train, y_train)
    y_pred_balanced = balanced.predict(X_test)

    # Best model (tuned C=10) — notebook Cell 70-71
    best_model = LogisticRegression(
        max_iter=1000, class_weight='balanced', C=10
    )
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]

    report_baseline = classification_report(y_test, y_pred_baseline, output_dict=True)
    report_balanced = classification_report(y_test, y_pred_balanced, output_dict=True)
    report_best     = classification_report(y_test, y_pred_best, output_dict=True)
    cm_best         = confusion_matrix(y_test, y_pred_best)
    roc_auc         = roc_auc_score(y_test, y_prob_best)
    fpr, tpr, _     = roc_curve(y_test, y_prob_best)

    return (best_model, vectorizer, df,
            report_baseline, report_balanced, report_best,
            cm_best, roc_auc, fpr, tpr, y_test, y_prob_best)


# ── Upload CSV ────────────────────────────────────────────────────────────────
st.sidebar.title("📂 Upload Dataset")
st.sidebar.markdown("Upload your `amazon_reviews.csv` to train the model.")
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV file", type=["csv"],
    help="Must contain 'reviewText' and 'overall' columns"
)

if uploaded_file is None:
    st.markdown("## 🛒 Amazon Review Sentiment Analyzer")
    st.info("👈 **Upload your `amazon_reviews.csv` in the sidebar to get started.**")
    st.markdown("""
    **Expected CSV columns:**
    - `reviewText` — the review text
    - `overall` — star rating (1–5)

    Ratings 4–5 → **Positive** | Ratings 1–3 → **Negative**
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", "4,914 reviews")
    col2.metric("Class Split", "90.5% / 9.5%")
    col3.metric("Expected AUC", "~0.93")
    st.stop()

with st.spinner("Loading data and training model..."):
    (model, vectorizer, df,
     rep_base, rep_bal, rep_best,
     cm, roc_auc, fpr, tpr,
     y_test, y_prob) = train_model(uploaded_file)

st.sidebar.success(f"✅ Loaded {len(df):,} reviews")
st.sidebar.metric("Positive", f"{(df['sentiment']==1).sum():,}")
st.sidebar.metric("Negative", f"{(df['sentiment']==0).sum():,}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Review Sentiment Analyzer")
st.caption("NLP Pipeline · TF-IDF (1,2-gram) · Logistic Regression · Class Imbalance Handling · Built by Shreevarsha S")

tabs = st.tabs([
    "🔍 Analyze Review",
    "📊 Model Performance",
    "🔬 Class Imbalance Fix",
    "📈 Dataset Insights"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Analyze an Amazon Product Review")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "Paste a product review:",
            placeholder="e.g. The memory card stopped working after one month, very disappointed...",
            height=130
        )

        with st.expander("💡 Try example reviews"):
            ex1, ex2 = st.columns(2)
            pos_ex = "This memory card works perfectly in my Samsung Galaxy. Fast delivery, great price, highly recommend!"
            neg_ex = "The card failed after just two weeks. Lost all my data. Very poor quality, would not recommend."
            if ex1.button("😊 Positive example"):
                user_input = pos_ex
                st.rerun()
            if ex2.button("😞 Negative example"):
                user_input = neg_ex
                st.rerun()

        analyze = st.button("Analyze Sentiment →", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Pipeline (from notebook):**")
        steps = [
            "① Lowercase + regex clean",
            "② Tokenize (split)",
            "③ Remove stopwords (NLTK)",
            "④ Lemmatize (WordNet)",
            "⑤ TF-IDF vectorize (1,2-gram)",
            "⑥ Logistic Regression predict"
        ]
        for s in steps:
            st.caption(s)
        st.info(f"**Best Model ROC-AUC:** {roc_auc:.3f}")

    if analyze and user_input.strip():
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        confidence = prob[pred] * 100

        st.divider()
        c1, c2, c3 = st.columns(3)

        label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
        css_class = "positive" if pred == 1 else "negative"

        with c1:
            st.markdown("**Sentiment**")
            st.markdown(f'<p class="{css_class}">{label}</p>', unsafe_allow_html=True)
        with c2:
            st.markdown("**Confidence**")
            st.progress(int(confidence))
            st.caption(f"{confidence:.1f}%")
        with c3:
            st.markdown("**Probabilities**")
            st.caption(f"Positive: {prob[1]*100:.1f}%")
            st.caption(f"Negative: {prob[0]*100:.1f}%")

        st.markdown("**Preprocessed text:**")
        st.code(cleaned if cleaned else "(empty after preprocessing)")

    elif analyze:
        st.warning("Please enter a review to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Model Performance — Best Model (C=10, class_weight='balanced')")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",        f"{rep_best['accuracy']:.0%}")
    m2.metric("ROC-AUC",         f"{roc_auc:.3f}")
    m3.metric("Negative Recall", f"{rep_best['0']['recall']:.0%}", "Was 16% before fix")
    m4.metric("Negative F1",     f"{rep_best['0']['f1-score']:.0%}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame({
            "Class":     ["Negative (0)", "Positive (1)"],
            "Precision": [rep_best['0']['precision'], rep_best['1']['precision']],
            "Recall":    [rep_best['0']['recall'],    rep_best['1']['recall']],
            "F1-Score":  [rep_best['0']['f1-score'],  rep_best['1']['f1-score']],
            "Support":   [int(rep_best['0']['support']), int(rep_best['1']['support'])]
        }).set_index("Class").round(2)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Pred Neg", "Pred Pos"],
                    yticklabels=["Actual Neg", "Actual Pos"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#3498db', lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1], 'k--', lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Best Model")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.info("**Notebook insight:** TF-IDF with bigrams (ngram_range=(1,2)) captures phrases like 'not good', 'stopped working' that unigrams miss.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Class Imbalance Fix (your key contribution)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Class Imbalance Fix — The Core Contribution")
    st.markdown("Dataset is **90.5% positive / 9.5% negative** — model ignores negatives without fixing.")

    col1, col2, col3 = st.columns(3)

    def show_model_card(col, title, report, highlight=False):
        with col:
            color = "#2ecc71" if highlight else "#3498db"
            st.markdown(f"**{title}**")
            neg_recall = report['0']['recall']
            pos_recall = report['1']['recall']
            acc = report['accuracy']
            st.metric("Negative Recall", f"{neg_recall:.0%}")
            st.metric("Positive Recall", f"{pos_recall:.0%}")
            st.metric("Accuracy",        f"{acc:.0%}")

    show_model_card(col1, "❌ Baseline (no fix)",     rep_base)
    show_model_card(col2, "✅ Balanced (class_weight)", rep_bal)
    show_model_card(col3, "🏆 Tuned (C=10)",           rep_best, highlight=True)

    st.divider()
    st.markdown("**Negative Recall Progression**")

    models  = ["Baseline\n(no fix)", "Balanced\n(class_weight)", "Tuned\n(C=10, best)"]
    recalls = [
        rep_base['0']['recall'],
        rep_bal['0']['recall'],
        rep_best['0']['recall']
    ]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(models, [r * 100 for r in recalls], color=colors,
                  edgecolor='white', width=0.5)
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{val:.0%}", ha='center', fontweight='bold', fontsize=13)
    ax.set_ylabel("Negative Recall (%)")
    ax.set_title("Negative Sentiment Recall — Before vs After Fix")
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"**Result:** Negative recall improved from {rep_base['0']['recall']:.0%} → {rep_best['0']['recall']:.0%} using class-weighted training — catching more dissatisfied customers.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dataset Insights
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Dataset Overview — Amazon Reviews (4,914 records)")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Reviews", "4,914")
    s2.metric("Positive (1)",  "4,449  (90.5%)")
    s3.metric("Negative (0)",  "465  (9.5%)")
    s4.metric("Features",      "5,000 TF-IDF")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Positive\n(rating 4-5)", "Negative\n(rating 1-3)"],
               [4449, 465], color=["#2ecc71", "#e74c3c"],
               edgecolor='white', width=0.5)
        ax.bar_label(ax.containers[0],
                     labels=["4,449 (90.5%)", "465 (9.5%)"],
                     padding=3, fontsize=11, fontweight='bold')
        ax.set_ylabel("Count")
        ax.set_title("90/10 Class Imbalance")
        ax.set_ylim(0, 5500)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Top Negative Keywords (from notebook)**")
        neg_kw = {
            'problem': 175, 'would': 171, 'work': 168,
            'month': 153, 'memory': 142, 'get': 140,
            'speed': 137, 'buy': 130, 'fail': 120, 'return': 115
        }
        neg_df = pd.DataFrame(list(neg_kw.items()), columns=['keyword', 'count'])
        neg_df = neg_df.sort_values('count')
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(neg_df['keyword'], neg_df['count'], color='#e74c3c', alpha=0.85)
        ax.set_title("Top Negative Sentiment Keywords")
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Key EDA Insights from Notebook**")
    i1, i2, i3 = st.columns(3)
    i1.info("**Negative reviews are longer**\nAvg 47.9 words vs 23.0 words for positive — unhappy customers write more")
    i2.info("**Top positive words**\ncard, work, phone, gb, great, memory, fast, price, sandisk")
    i3.info("**Top negative words**\nproblem, month, work, speed, fail — time-based failures are common")
