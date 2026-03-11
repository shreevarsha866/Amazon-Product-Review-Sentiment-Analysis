# 🗨️ Amazon Product Review Sentiment Analysis

> **NLP Classification Pipeline — Handling Real-World Class Imbalance**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?logo=python)](https://nltk.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyR0D_iIqua-wm-nZMac982I_llFTYoY?usp=sharing)

---

## 📌 Project Overview

This project builds an end-to-end **NLP sentiment classification pipeline** for 4,914 Amazon product reviews — classifying each as **Positive** or **Negative**. The core challenge was a severe **90.5% / 9.5% class imbalance** that caused the baseline model to miss nearly all negative reviews.

By applying **class weighting and hyperparameter tuning**, negative sentiment recall improved from **16% → 72%** while maintaining **92% overall accuracy**. An interactive **Streamlit dashboard** was built for real-time review prediction and model visualization.

---

## 🎯 Business Problem

A model that only detects positive reviews is dangerous — it hides dissatisfaction and leads to poor product decisions. This project solves that by building a **balanced, business-aware classifier** that reliably catches unhappy customers even in heavily skewed data.

---

## 📊 Key Results — All Models (From Notebook)

### Model Comparison

| Model | Accuracy | Neg Precision | Neg Recall | Neg F1 | Pos Recall |
|---|---|---|---|---|---|
| LR Baseline (no fix) | 92% | 0.94 | **16%** ❌ | 0.28 | 100% |
| Multinomial Naive Bayes | 91% | 0.80 | **4%** ❌❌ | 0.08 | 100% |
| LR Balanced (class_weight) | 91% | 0.57 | **71%** ✅ | 0.63 | 94% |
| **LR Tuned (C=10) — Best** | **93%** | **0.62** | **61%** ✅ | **0.62** | **96%** |

### Streamlit Dashboard Results (Real CSV)

| Metric | Value |
|---|---|
| **Accuracy** | 93% |
| **ROC-AUC** | **0.928** |
| **Negative Recall** | 61% (was 16%) |
| **Negative F1** | 0.62 |
| **Positive Recall** | 96% |

### Confusion Matrix — Best Model (C=10)

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 57 ✅ | 36 |
| **Actual Positive** | 35 | 855 ✅ |

> ✅ **Final Model: LR (class_weight='balanced', C=10)** — Best balance of accuracy and negative recall. ROC-AUC: **0.928**

---

## 🗂️ Dataset

- **Source:** `amazon_reviews.csv` — Amazon product reviews (memory cards)
- **Total records:** 4,915 raw → **4,914 after null drop**
- **Columns used:** `reviewText`, `overall`
- **Duplicates:** 0

### Rating Distribution (raw)

| Rating | Count |
|---|---|
| ⭐⭐⭐⭐⭐ (5) | 3,922 |
| ⭐⭐⭐⭐ (4) | 527 |
| ⭐⭐⭐ (3) | 142 |
| ⭐⭐ (2) | 80 |
| ⭐ (1) | 244 |

### Sentiment Mapping (Target Variable)

| Ratings | Label | Count | % |
|---|---|---|---|
| 4–5 stars | Positive (1) | 4,449 | **90.5%** |
| 1–3 stars | Negative (0) | 466 | **9.5%** |

---

## 🏗️ Project Pipeline

```
Raw Data (amazon_reviews.csv — 4,915 rows, 12 columns)
     │
     ▼
1. Data Inspection
   → shape (4915, 12), dtypes, null check, duplicate check (0 found)
   → overall value counts: 5★(3922), 4★(527), 1★(244), 3★(142), 2★(80)
     │
     ▼
2. Target Variable Creation
   → ratings 1-3 → Negative (0) | ratings 4-5 → Positive (1)
   → 4,449 positive (90.5%) | 466 negative (9.5%)
     │
     ▼
3. Data Cleaning
   → keep only [reviewText, sentiment]
   → drop 1 null → 4,914 clean records
     │
     ▼
4. Text Preprocessing
   → lowercase + regex clean [^a-zA-Z\s]
   → tokenize (split)
   → stopword removal (NLTK English)
   → lemmatization (WordNetLemmatizer)
   → join back → final_text
     │
     ▼
5. EDA
   → avg review length: negative=47.9 words vs positive=23.0 words
   → top positive words: card, work, phone, gb, great, memory, fast
   → top negative words: card, problem, would, work, month, speed
     │
     ▼
6. Feature Engineering
   → TF-IDF (max_features=5000, ngram_range=(1,2))
   → Shape: (4914, 5000)
   → Sample features: 'able', 'able copy', 'able find', 'absolutely issue'
     │
     ▼
7. Model Training
   → Train-Test Split: 80/20, stratified, random_state=42
   → Baseline LR → Balanced LR → Naive Bayes → Tuned LR (GridSearchCV)
     │
     ▼
8. Hyperparameter Tuning
   → GridSearchCV: C=[0.01, 0.1, 1, 10], cv=5, scoring='f1'
   → Best: C=10
     │
     ▼
9. Streamlit Dashboard
   → Live review prediction + model performance + imbalance visualization
```

---

## 🧹 Text Preprocessing Pipeline

```python
# Step 1 — Lowercase & Regex Clean
df['cleaned_text'] = df['reviewText'].str.lower()
df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Step 2 — Tokenization
df['tokens'] = df['cleaned_text'].str.split()

# Step 3 — Stopword Removal
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(
    lambda words: [w for w in words if w not in stop_words]
)

# Step 4 — Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(
    lambda words: [lemmatizer.lemmatize(w) for w in words]
)

# Step 5 — Join Back
df['final_text'] = df['tokens'].apply(lambda words: ' '.join(words))
```

---

## ⚙️ Feature Engineering — TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)   # unigrams + bigrams
)
X_tfidf = vectorizer.fit_transform(df['final_text'])
# Shape: (4914, 5000)
```

Sample TF-IDF features: `'able'`, `'able copy'`, `'able find'`, `'absolutely issue'`, `'accept'`

---

## 🤖 Model Training & Imbalance Handling

### Why Baseline Failed

```
Baseline Confusion Matrix (no fix):
  True Negatives  (caught):  15 out of 93 ← only 16% caught ❌
  False Negatives (missed):  78 out of 93 ← 84% of complaints MISSED ❌
→ Model learned to predict everything as positive (majority class)
```

### Fix — class_weight='balanced'

```python
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Confusion Matrix after fix:
# True Negatives:  67  (was 15) ✅
# Negative Recall: 71% (was 16%) ✅
```

### Hyperparameter Tuning

```python
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid, cv=5, scoring='f1'
)
grid.fit(X_train, y_train)
# Best: C=10
```

---

## 📈 Negative Recall Progression

| Stage | Negative Recall | Change |
|---|---|---|
| Baseline (no fix) | 16% | — |
| Balanced (class_weight) | 71% | +55% ✅ |
| Tuned (C=10) | 61% | Best accuracy + recall tradeoff ✅ |

---

## 💡 Key EDA Insights (From Notebook)

| Insight | Detail |
|---|---|
| **Review length** | Negative reviews avg **47.9 words** vs positive **23.0 words** — unhappy customers write more |
| **Top positive words** | card (4298), work (1876), phone (1553), gb (1469), great (1389), memory (1236), fast (861) |
| **Top negative words** | card (1239), problem (175), would (171), work (168), month (153), speed (137) |
| **Class imbalance** | 90.5% positive / 9.5% negative — model ignores minority without fix |
| **Naive Bayes failure** | 4% negative recall — predicted nearly every review as positive |

---

## 🖥️ Streamlit Dashboard

An interactive dashboard was built for real-time analysis:

**Tab 1 — Analyze Review**
- Paste any review → runs full preprocessing pipeline → predicts sentiment
- Shows confidence %, probability scores, preprocessed text

**Tab 2 — Model Performance**
- Classification report, confusion matrix, ROC curve (AUC: 0.928)

**Tab 3 — Class Imbalance Fix**
- Side-by-side comparison: Baseline (16%) → Balanced (71%) → Tuned (61%)
- Visual bar chart showing recall progression

**Tab 4 — Dataset Insights**
- Class distribution chart, top negative keywords bar chart, EDA stats

```bash
# Run dashboard
pip install streamlit scikit-learn nltk pandas matplotlib seaborn
streamlit run sentiment_app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core programming |
| **Pandas & NumPy** | Data manipulation |
| **NLTK** | Tokenization, stopword removal, lemmatization |
| **Scikit-learn** | TF-IDF, Logistic Regression, GridSearchCV, Naive Bayes |
| **Matplotlib & Seaborn** | EDA visualizations |
| **Streamlit** | Interactive dashboard |
| **Google Colab** | Development environment |

---

## 🚀 How to Run

**Option 1 — Google Colab (Notebook)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyR0D_iIqua-wm-nZMac982I_llFTYoY?usp=sharing)

**Option 2 — Streamlit Dashboard**
```bash
git clone https://github.com/shreevarsha866/Amazon-Product-Review-Sentiment-Analysis
cd Amazon-Product-Review-Sentiment-Analysis
pip install -r requirements.txt
streamlit run sentiment_app.py
# Upload amazon_reviews.csv in the sidebar
```

**Option 3 — Run Locally (Notebook)**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
jupyter notebook Amazon_reviews_for_sentiment_analysis.ipynb
```

---

## 📁 Repository Structure

```
📦 Amazon-Product-Review-Sentiment-Analysis
 ┣ 📓 Amazon_reviews_for_sentiment_analysis.ipynb  ← Main notebook (Colab)
 ┣ 🖥️ sentiment_app.py                             ← Streamlit dashboard
 ┣ 📄 requirements.txt                             ← Python dependencies
 ┗ 📖 README.md
```

> Note: `amazon_reviews.csv` is not included — upload via the Streamlit dashboard.

---

## 💡 Conclusion

- **TF-IDF + bigrams** captures phrases like "not good", "stopped working" — critical for NLP
- **Class imbalance (90/10) is the biggest real-world NLP challenge** — solved using class weighting
- **Naive Bayes completely failed** (4% recall) — model selection matters significantly
- **Negative recall improved 16% → 71%** using class_weight='balanced'
- **ROC-AUC 0.928** on real data — strong performance for a linear model

---

## 🔮 Future Work

- Try SVM, XGBoost, BERT for improved performance
- Apply SMOTE oversampling as alternative imbalance strategy
- Build FastAPI for real-time review classification
- Extend to multi-class sentiment (positive / neutral / negative)
- Deploy Streamlit app on Streamlit Cloud

---

## 👩‍💻 Author

**Shreevarsha S** — Data Scientist | ML · NLP Enthusiast

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-7b6ef6)](https://shreevarsha866.github.io/Shreevarsha_Portfolio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/s-shreevarsha-503887218/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shreevarsha866)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:varshashree866@gmail.com)

---
*⭐ If you found this project helpful, please give it a star!*
