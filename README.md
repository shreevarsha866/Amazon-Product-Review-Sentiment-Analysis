# 🗨️ Amazon Product Review Sentiment Analysis

> **NLP Classification Pipeline — Handling Real-World Class Imbalance**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?logo=python)](https://nltk.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyR0D_iIqua-wm-nZMac982I_llFTYoY?usp=sharing)

---

## 📌 Project Overview

This project builds an end-to-end **NLP sentiment classification pipeline** for 4,914 Amazon product reviews — classifying each review as **Positive** or **Negative**. The core challenge was a severe **90.5% / 9.5% class imbalance** that caused the baseline model to miss nearly all negative reviews.

By applying **class weighting and hyperparameter tuning**, negative sentiment recall was improved from **22% → 72%** while maintaining **92% overall accuracy** — making the model genuinely useful for real business decisions.

---

## 🎯 Business Problem

Customer reviews contain critical signals about product quality and customer satisfaction. A model that **only detects positive reviews** is dangerous — it hides dissatisfaction and leads to poor product decisions.

This project solves that by building a **balanced, business-aware classifier** that reliably catches unhappy customers even in heavily skewed data.

---

## 📊 Key Results

| Model | Accuracy | Negative Recall | F1 (Negative) |
|---|---|---|---|
| Logistic Regression (Baseline) | 92% | 22% ❌ | 0.35 |
| Multinomial Naive Bayes | — | **0%** ❌❌ | 0.00 |
| Logistic Regression (Balanced) | 91% | **72%** ✅ | 0.61 |
| **Logistic Regression (Tuned C=10)** | **92%** | 63% ✅ | **0.60** |

> ✅ **Final Model: Balanced Logistic Regression — 92% accuracy, 72% negative recall, 0.95 F1-score overall**

---

## 🗂️ Project Pipeline

```
Raw Data (amazon_reviews.csv)
     │
     ▼
1. Data Inspection       → shape, dtypes, nulls, duplicates, class distribution
     │
     ▼
2. Target Variable       → map ratings 1-3 → Negative (0), 4-5 → Positive (1)
     │
     ▼
3. Data Cleaning         → drop nulls, keep reviewText + sentiment
     │
     ▼
4. Text Preprocessing    → lowercase → regex clean → tokenize →
                           stopword removal → lemmatization
     │
     ▼
5. EDA                   → sentiment distribution, review length,
                           most frequent positive/negative words
     │
     ▼
6. Feature Engineering   → TF-IDF Vectorization (top 5000 features)
     │
     ▼
7. Model Training        → Logistic Regression (baseline → balanced → tuned)
                           + Multinomial Naive Bayes (comparison)
     │
     ▼
8. Evaluation            → Classification report, Confusion Matrix
     │
     ▼
9. Hyperparameter Tuning → GridSearchCV (C: 0.01, 0.1, 1, 10) with 5-fold CV
```

---

## 📂 Dataset

- **Source:** Amazon Product Reviews (`amazon_reviews.csv`)
- **Size:** 4,914 reviews
- **Class Distribution:**

| Sentiment | Count | Percentage |
|---|---|---|
| Positive (4–5 stars) | 4,447 | **90.5%** |
| Negative (1–3 stars) | 467 | **9.5%** |

> ⚠️ This extreme imbalance is the core challenge of this project.

---

## 🧹 Text Preprocessing Pipeline

```python
# Step 1 — Lowercase & clean
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

# Step 5 — Join back
df['final_text'] = df['tokens'].apply(lambda words: ' '.join(words))
```

---

## ⚙️ Feature Engineering — TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['final_text'])

# Result: 4914 reviews × 5000 features
```

---

## 🤖 Model Training & Imbalance Handling

### Problem — Baseline model missed 73 out of 93 negative reviews:
```
Confusion Matrix (Baseline):
  True Negatives:  20  ← only 20 negatives caught ❌
  False Negatives: 73  ← 73 negatives missed ❌
```

### Solution — class_weight='balanced':
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Result:
# True Negatives:  67  ← was 20 ✅
# Negative Recall: 72% ← was 22% ✅
```

### Hyperparameter Tuning with GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid, cv=5, scoring='f1'
)
grid.fit(X_train, y_train)
# Best: C=10
```

---

## 📈 Model Comparison

| Model | Accuracy | Neg Recall | What went wrong |
|---|---|---|---|
| Logistic Regression (default) | 92% | 22% | Biased toward majority class |
| Multinomial Naive Bayes | ~90% | **0%** | Predicted ALL reviews as positive |
| **Logistic Regression (balanced)** | **91%** | **72%** ✅ | Best for catching unhappy customers |
| Logistic Regression (tuned C=10) | 92% | 63% | Slightly better accuracy, lower recall |

> ✅ **Balanced model chosen as final** — better at catching dissatisfied customers, which is the real business goal

---

## 💡 Key Business Insights

- ⭐ **Positive reviews** most frequently contain: *great, love, perfect, best, easy*
- ❌ **Negative reviews** most frequently contain: *return, waste, poor, broke, disappointed*
- 📏 **Negative reviews are longer** on average — dissatisfied customers write more
- ⚠️ A model with **22% negative recall is dangerous** — it hides 78% of customer complaints
- ✅ After balancing: model now **detects 72% of unhappy customers** — actionable for business

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core programming |
| **Pandas & NumPy** | Data manipulation |
| **NLTK** | Tokenization, stopword removal, lemmatization |
| **Scikit-learn** | TF-IDF, Logistic Regression, GridSearchCV |
| **Matplotlib & Seaborn** | EDA visualizations |
| **Google Colab** | Development environment |

---

## 🚀 How to Run

**Option 1 — Google Colab (Recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyR0D_iIqua-wm-nZMac982I_llFTYoY?usp=sharing)

**Option 2 — Run Locally**

```bash
# Clone the repo
git clone https://github.com/shreevarsha866/Amazon-Product-Review-Sentiment-Analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run the notebook
jupyter notebook Amazon_reviews_for_sentiment_analysis.ipynb
```

---

## 📁 Repository Structure

```
📦 Amazon-Product-Review-Sentiment-Analysis
 ┣ 📓 Amazon_reviews_for_sentiment_analysis.ipynb  ← Main notebook
 ┣ 📄 amazon_reviews.csv                           ← Dataset
 ┗ 📖 README.md
```

---

## 💡 Conclusion

- TF-IDF + Logistic Regression is a strong, interpretable NLP baseline
- **Class imbalance is the biggest real-world NLP challenge** — solved here using class weighting
- Naive Bayes completely failed (0% negative recall) — model selection matters
- The final model balances accuracy with business utility — catching dissatisfied customers reliably

---

## 🔮 Future Work

- Try advanced models — SVM, XGBoost, BERT for improved performance
- Apply SMOTE oversampling as an alternative imbalance strategy
- Build a real-time review classification API using FastAPI
- Extend to multi-class sentiment (positive / neutral / negative)

---

## 👩‍💻 Author

**Shreevarsha S**
Data Science Professional | ML & NLP Enthusiast

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-7b6ef6?logo=globe)](https://shreevarsha866.github.io/Shreevarsha_Portfolio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/s-shreevarsha-503887218/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shreevarsha866)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:varshashree866@gmail.com)

---

*⭐ If you found this project helpful, please give it a star!*
