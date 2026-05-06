# 🖱️ Online Ad Engagement Prediction — Machine Learning Approach

> Predicting whether a user will click on an online advertisement based on cursor movement patterns, session behavior, and ad metadata — using Decision Tree, Random Forest, and MLP classifiers.

---

## 📌 Project Overview

This project builds a machine learning pipeline to predict ad click behavior from real user interaction data. By analyzing how users move their cursors, scroll, and engage with a webpage, the models estimate the probability of a user clicking on an online advertisement.

The dataset combines user interaction logs, demographic metadata, and ad-related details into a unified pipeline — from raw `.tsv` and `.csv` files all the way to trained, evaluated classifiers.

---

## 📂 Dataset — Attentive Cursor Dataset

| File | Description |
|---|---|
| `groundtruth.tsv` | Whether each user clicked an ad, plus session log ID |
| `participants.tsv` | Country, ad type, position, category, and query |
| `logs/` | Per-session CSV files with cursor positions, timestamps, and events |

**Key facts:**
- 2,909 merged records across participants and sessions
- Filtered to 2,500 users who spent ≥ 5 seconds on the page
- 74.4% non-clicks / 25.6% clicks (class imbalance present)
- 69 countries represented — dropped due to high cardinality

---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing
- Merged `groundtruth.tsv` and `participants.tsv` on `log_id`
- Dropped high-cardinality and restricted columns (`user_id`, `attention`, `serp_id`, `query`, `country`)
- Removed duplicates

### 2. Feature Engineering (from log files)
Three behavioral features were extracted by looping through all session CSVs:

| Feature | Description |
|---|---|
| `duration` | Total time spent on the page (max − min timestamp in seconds) |
| `average_speed` | Mean cursor speed across the session (pixels/second) |
| `clicks_per_session` | Number of click events recorded |
| `scroll_activity_per_session` | Number of scroll events recorded |

### 3. Category Simplification
14 ad categories were grouped into 4 broader classes to reduce cardinality:

| New Category | Original Categories |
|---|---|
| Shop | Apparel, Luxury Goods, Toys, Sporting Goods, etc. |
| Technology | Computers & Electronics, Games |
| Hobby | Travel, Food & Drink |
| Finance | Autos & Vehicles, Real Estate |

### 4. Preprocessing Pipeline
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `SimpleImputer` for missing values (mean / most frequent)
- `np.inf` replaced with `NaN` before imputation
- Train / Validation / Test split: **70% / 15% / 15%**

### 5. Noise Augmentation
- Training set doubled with a noisy copy (5% of each feature's std)
- Log-transformed versions of numeric features added
- Outliers clipped before augmentation

---

## 🧠 Models

### Decision Tree (Baseline)
```python
GridSearchCV(DecisionTreeClassifier(class_weight='balanced'),
    param_grid={
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [2, 4, 6],
        'min_samples_split': [5, 10, 15],
        'ccp_alpha': [0.0, 0.01, 0.05]
    }, cv=5, scoring='f1_weighted')
```

### Random Forest
```python
RandomizedSearchCV(RandomForestClassifier(),
    param_distributions={
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'class_weight': ['balanced', 'balanced_subsample', None],
        ...
    }, n_iter=50, cv=3, scoring='f1_weighted')
```

### MLP Classifier (Grid Search + Randomized Search)
```python
GridSearchCV(MLPClassifier(),
    param_grid={
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }, cv=2, scoring='f1_weighted')
```

---

## 📊 Results

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Decision Tree | 0.7562 | 0.6640 | 0.6872 | 0.6640 |
| Random Forest | 0.7575 | 0.7173 | 0.7311 | 0.7173 |
| **MLP — Grid Search** | **0.7646** | **0.7867** | **0.7636** | **0.7867** ✅ |
| MLP — Randomized Search | 0.7638 | 0.7867 | 0.7549 | 0.7867 |

**Winner: MLP Classifier (Grid Search)** — best F1 and recall across all models.

> Full results, confusion matrices, and ROC curves are available in the notebook.

---

## 💡 Key Insights

- **Cursor speed, session duration, and scroll activity** were the most predictive behavioral features.
- **Category simplification** improved model generalizability by reducing noise from high-cardinality categorical columns.
- **Class imbalance** (74/26 split) was partially addressed through class weighting and noise augmentation — but remains a challenge for minimizing false negatives.
- **MLP captured nonlinear interactions** between behavioral and contextual features that tree-based models couldn't fully leverage.
- **SHAP values** and additional feature engineering are recommended for future improvement.

---

## 🛠 Tech Stack

```
Python 3.10
├── pandas / numpy
├── scikit-learn
│   ├── DecisionTreeClassifier
│   ├── RandomForestClassifier
│   ├── MLPClassifier
│   ├── GridSearchCV / RandomizedSearchCV
│   ├── StandardScaler / OneHotEncoder / SimpleImputer
│   └── roc_curve / auc / confusion_matrix
├── matplotlib
└── Google Colab + Drive
```

---

## 📁 Repository Structure

```
online-ad-engagement-prediction/
│
├── notebook.ipynb              # Full pipeline notebook
├── README.md                   # This file
└── data/                       # Dataset (not included — see below)
    ├── groundtruth.tsv
    ├── participants.tsv
    └── logs/
```

> **Note:** The Attentive Cursor Dataset is not included in this repository. It was accessed via Google Drive during development.

---

## 🔮 Future Work

- Apply **SMOTE** or cost-sensitive learning to better handle class imbalance
- Incorporate **SHAP values** for model explainability
- Explore **XGBoost / LightGBM** as stronger ensemble alternatives
- Add richer features: cursor trajectory patterns, hover duration, viewport position

---

## 👤 Author

**Seyed Danial Maktabi**
MSc Data Analytics | Mechatronics Engineer
Budapest, Hungary

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/danial-maktabi-79aab118a/)

---

## 📄 License

This project was developed as an MSc dissertation.
