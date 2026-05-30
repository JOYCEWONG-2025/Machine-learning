# 💘 Tying the (Data) Knot: Love, Life & Likes

> **WIA1006 / WID3006 Machine Learning** — Group Assignment  
> Sem 2, Session 2025/2026 · Faculty of Computer Science & Information Technology, Universiti Malaya

A machine learning project that predicts dating app match outcomes — **Ghosted**, **Mutual Match**, or **Catfished** — using behavioral and demographic features from a synthetic dating app dataset.

🔗 **[Live Interactive Dashboard →]((https://dating-predictor-ml.netlify.app/))**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [The Situationship Index](#-the-situationship-index-innovation-layer)
- [Model Results](#-model-results)
- [Visualizations](#-visualizations)
- [Interactive Web App](#-interactive-web-app)
- [How to Run](#-how-to-run)
- [Key Findings](#-key-findings)
- [Group Members](#-group-members)

---

## 🔍 Project Overview

Modern dating app interactions generate rich behavioral data — swipe patterns, messaging frequency, session durations, profile completeness — yet predicting relationship outcomes remains challenging. This project investigates whether machine learning can classify match outcomes using purely behavioral signals.

**Target classes:**

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Ghosted | One party ceases communication without explanation |
| 1 | Mutual Match | Both parties express reciprocal interest |
| 2 | Catfished | User interacted with a deceptive/fraudulent profile |

**Key innovation:** We designed the **Situationship Index**, a composite behavioral metric (0–100) that quantifies non-committal engagement patterns in digital dating.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset) |
| Records | 50,000 (synthetic) |
| Features | 19 |
| After filtering | 14,974 records · 3 balanced classes |
| Type | Multi-class classification |

**Features used:**
- `AppUsage` — normalized app usage time
- `SwipeRatio` — swipe right ratio
- `message_sent_count` — messaging activity
- `profile_pics_count` — number of profile photos
- `bio_length` — profile bio completeness
- `emoji_usage_rate` — emoji usage in messages
- `location_type` — Urban (1) / Rural (0)
- `education_level` — ordinal scale 1–4
- `Situationship_Index` — engineered composite feature

---

## 📁 Project Structure

```
📦 dating-app-ml/
├── 📓 notebook/
│   └── dating_app_behavior.ipynb     # Full Kaggle notebook
├── 📊 charts/
│   ├── chart1_correlation_heatmap.png
│   ├── chart2_index_distribution.png
│   ├── chart3_boxplot_index_vs_outcome.png
│   ├── chart4_feature_importance.png
│   ├── chart5_model_comparison.png
│   └── chart6_confusion_matrices.png
├── 🌐 app/
│   └── dating_predictor.html         # Standalone interactive dashboard
├── 📄 report/
│   └── ML_Group_Report_Final.docx    # Full project report
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Preprocessing
- Missing values imputed using column mode
- Column names stripped of whitespace
- Six behavioral features normalized with `MinMaxScaler`

### 2. Encoding
- `location_type`: Urban = 1, Rural = 0
- `education_level`: ordinal scale with expanded variant mapping  
  *(handles: High School, Diploma, Undergraduate, Bachelor's, Postgraduate, Master's, MBA, PhD, Postdoc)*
- Other categoricals: label-encoded via pandas `.cat.codes`

### 3. Feature Engineering
- Engineered **Situationship Index** (see section below)
- Final feature set: 8 base features + Situationship Index

### 4. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Followed by `StandardScaler` normalization.

### 5. Models Trained
Five classifiers from distinct algorithmic families:

| Model | Family | Key Config |
|-------|--------|------------|
| Logistic Regression | Linear | `max_iter=3000, class_weight='balanced'` |
| Random Forest | Ensemble (Bagging) | `n_estimators=100, random_state=42` |
| Gradient Boosting | Ensemble (Boosting) | Default sklearn params |
| KNN | Instance-Based | Default `k=5` |
| SVM | Kernel-Based | Default RBF kernel |

### 6. Hyperparameter Tuning
`GridSearchCV` applied to Random Forest:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
# Best: max_depth=None, min_samples_split=5, n_estimators=200
```

### 7. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrices (top 2 models)
- 5-fold cross-validation
- AutoML-style comparison (Naive Bayes, LDA, Decision Tree)

---

## 💡 The Situationship Index (Innovation Layer)

A novel composite behavioral metric designed to quantify non-committal engagement tendencies on a 0–100 scale.

**Formula:**

```
Situationship Index = 100 × (
    0.35 × AppUsage +
    0.25 × SwipeRatio +
    0.25 × (1 − MatchRate) +
    0.15 × (1 − Efficiency)
)
```

Where:
- `MatchRate = mutual_matches / (likes_received + 1)`
- `Efficiency = mutual_matches / (AppUsage + 0.1)`

**Score interpretation:**

| Score Range | Behavior Type | Description |
|-------------|---------------|-------------|
| 0 – 29 | Committed | Low app investment, high match efficiency |
| 30 – 44 | Intentional | Selective engagement, good conversion |
| 45 – 59 | Casual | Moderate usage, mixed match rates |
| 60 – 74 | Avoidant | High investment, low returns |
| 75 – 100 | Situationship | Maximum app time, minimal efficiency |

**Impact:**
- Improved Logistic Regression accuracy: 33.32% → 33.96% (+0.64%)
- Ranked **#1 in Random Forest feature importance** (0.172)
- Index correlation with target: ~0.008 *(consistent with synthetic data ceiling)*

---

## 📈 Model Results

### Test Accuracy vs Cross-Validation

| Rank | Model | Test Accuracy | CV Mean (5-Fold) | CV Std |
|------|-------|--------------|-----------------|--------|
| 🥇 1 | Logistic Regression (w/ Index) | **33.96%** | 33.10% | ±0.46% |
| 🥈 2 | Random Forest (Baseline) | 33.76% | 32.86% | ±0.48% |
| 🥉 3 | SVM | 33.02% | 33.40% | ±0.62% |
| 4 | Gradient Boosting | 32.85% | 33.50% | ±0.31% |
| 5 | KNN | 32.49% | 33.35% | ±0.59% |
| — | Random Forest (Tuned) | 31.72% | — | — |
| — | Random Baseline (3-class) | 33.33% | — | — |

### AutoML-Style Comparison

| Model | Accuracy | Type |
|-------|----------|------|
| Naive Bayes | 34.32% | AutoML-style extra |
| Logistic Regression | 33.96% | Our model |
| LDA | 33.92% | AutoML-style extra |
| Random Forest | 33.76% | Our model |
| SVM | 33.02% | Our model |
| Gradient Boosting | 32.85% | Our model |
| KNN | 32.49% | Our model |
| Decision Tree | 32.49% | AutoML-style extra |

> **Conclusion:** Even AutoML-style search peaks at ~34.3%, confirming a hard information ceiling imposed by the synthetic dataset.

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Situationship Index ⭐ | 17.2% |
| 2 | Bio length | 16.4% |
| 3 | App usage | 15.8% |
| 4 | Messages sent | 14.7% |
| 5 | Emoji usage | 14.2% |
| 6 | Swipe ratio | 14.2% |
| 7 | Profile pics | 7.6% |

---

## 📊 Visualizations

| Chart | Description |
|-------|-------------|
| Figure 1 | Correlation heatmap of all features |
| Figure 2 | Distribution of Situationship Index scores |
| Figure 3 | Boxplot — Situationship Index by match outcome |
| Figure 4 | Feature importance (Random Forest) |
| Figure 5 | Model accuracy comparison (all 5 models) |
| Figure 6 | Confusion matrices — top 2 models |

> All charts generated in Kaggle notebook and saved as `.png` files.

---

## 🌐 Interactive Web App

A fully deployed interactive dashboard built with HTML/CSS/JavaScript + Chart.js.

**🔗 Live URL: [https://incomparable-beignet-c6453e.netlify.app/](https://incomparable-beignet-c6453e.netlify.app/)**

**Features:**
- 🎚️ Behavioral sliders for all 6 input features
- 💘 Real-time outcome prediction (weighted ensemble of 5 models)
- 📊 Live Situationship Index score with behavior type label
- 📉 Class probability bars (Mutual Match / Ghosted / Catfished)
- 🤖 Per-model prediction table showing all 5 models individually
- 📈 Dataset insight tabs: model comparison, feature importance, index distribution

**Deployed via:** Netlify (free static hosting)

---

## 🚀 How to Run

### Option 1 — Kaggle (Recommended)
1. Open the notebook on Kaggle
2. Attach the dataset: `keyushnisar/dating-app-behavior-dataset`
3. Click **Run All**

### Option 2 — Local
```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

# Run the notebook
jupyter notebook dating_app_behavior.ipynb
```

**Requirements:**
```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🔑 Key Findings

1. **Data ceiling is real** — All 5 models cluster within ~1% of the 33.33% random baseline, confirmed by AutoML comparison peaking at 34.32%. The synthetic dataset assigns labels independently of features.

2. **Situationship Index adds value** — Ranks #1 in feature importance and contributes a consistent +0.64% accuracy gain, demonstrating that domain-informed feature engineering works even under constrained conditions.

3. **Logistic Regression wins** — Achieves highest test accuracy (33.96%) with the most balanced per-class performance. More complex models (Gradient Boosting, Random Forest) did not outperform the linear baseline.

4. **Tuning can backfire** — GridSearchCV tuned RF (31.72%) underperformed the baseline RF (33.76%), illustrating that hyperparameter tuning can overfit to noise when no real signal exists.

5. **Gradient Boosting is most stable** — Highest CV mean (33.50%) with lowest variance (±0.31%), making it the safest choice for production deployment.

---

## 👥 Group Members

| Name | Student ID |
|------|-----------|
| [Member 1] | [ID] |
| [Member 2] | [ID] |
| [Member 3] | [ID] |
| [Member 4] | [ID] |
| [Member 5] | [ID] |

---

## 📚 References

- Kaggle Dataset: keyushnisar. (2024). *Dating App Behavior Dataset*. https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Feurer, M., et al. (2015). Efficient and Robust Automated Machine Learning. *NeurIPS*, 28.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.

---

<div align="center">
  <sub>WIA1006 / WID3006 Machine Learning · Sem 2, Session 2025/2026 · Universiti Malaya</sub>
</div>
