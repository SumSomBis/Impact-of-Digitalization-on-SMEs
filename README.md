# 🏪 Digital Adoption Prediction for Small Businesses

> A machine learning project to predict whether small shops and local businesses are likely to adopt digital tools, using survey-based data and ensemble classification models.

---

## 📌 Overview

Many small businesses are slow to adopt digital tools despite the growing need for digital transformation. This project builds a binary classification pipeline that predicts a shop's likelihood of digital adoption based on survey responses — enabling targeted outreach, policy planning, and support programs.

A shop is classified as a **Likely Adopter** if they rated their intent to use more digital tools as **4 or above** (on a 1–5 scale), and **Unlikely Adopter** otherwise.

---

## 📁 Project Structure

```
├── FinalModel.py          # Full pipeline with SMOTE, feature selection & GridSearchCV
├── FinalModelAna.py       # Baseline pipeline without SMOTE (analysis version)
├── Final.csv              # Survey dataset (required)
├── README.md
```

---

## 🔄 Two Model Pipelines

### `FinalModel.py` — Advanced Pipeline
- Correlation-based feature selection (|corr| ≥ 0.2 with target)
- SMOTE oversampling to handle class imbalance
- Soft voting ensemble
- GridSearchCV to tune ensemble voting weights

### `FinalModelAna.py` — Baseline Pipeline
- All features used (no correlation filter)
- No SMOTE (original class distribution)
- Hard voting ensemble
- Manual best-model selection by test accuracy

---

## ⚙️ Pipeline Walkthrough

```
Raw Survey CSV
      │
      ▼
Drop non-numeric columns (Location, Shop Name)
      │
      ▼
Create target variable: Digital Adoption (0 or 1)
      │
      ▼
Convert all features to numeric → Drop NaN rows
      │
      ▼
[FinalModel only] Correlation-based feature selection
      │
      ▼
Train/Test Split (80/20, stratified)
      │
      ▼
[FinalModel only] Apply SMOTE on training data
      │
      ▼
Train: Random Forest | Decision Tree | Naive Bayes | Voting Ensemble
      │
      ▼
Evaluate: Accuracy, Classification Report, Confusion Matrix
      │
      ▼
[FinalModel only] GridSearchCV to optimize ensemble weights
      │
      ▼
Generate predictions + adoption probabilities on full dataset
      │
      ▼
Export results → Final_with_predictions_and_summary.csv
```

---

## 🤖 Models Used

| Model | Library | Notes |
|---|---|---|
| Random Forest | `sklearn.ensemble` | 200 estimators, `random_state=42` |
| Decision Tree | `sklearn.tree` | `random_state=42` |
| Gaussian Naive Bayes | `sklearn.naive_bayes` | — |
| Voting Ensemble | `sklearn.ensemble` | Soft voting (`FinalModel`) / Hard voting (`FinalModelAna`) |

---

## 📊 Output

Both scripts generate a CSV with the following columns appended to the original data:

| Column | Description |
|---|---|
| `Predicted Adoption` | `1` = Likely Adopter, `0` = Unlikely Adopter |
| `Adoption Probability (%)` | Model confidence score (0–100%) |
| `Adoption Category` | Human-readable label |
| `Total Shops Analyzed` | Summary statistic |
| `Total Likely Adopters` | Summary statistic |
| `Overall Adoption Rate (%)` | Summary statistic |
| `Model Used` | Best model name |
| `Model Accuracy (%)` | Overall model accuracy |

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `scikit-learn` | ML models, metrics, preprocessing |
| `imbalanced-learn` | SMOTE oversampling (`FinalModel.py` only) |
| `matplotlib`, `seaborn` | Visualization |

---

## 🚀 Usage

1. Clone the repository and place `Final.csv` in the root directory.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the advanced pipeline:

```bash
python FinalModel.py
```

4. Or run the baseline analysis:

```bash
python FinalModelAna.py
```

Output files will be saved in the same directory:
- `Final_with_predictions_and_summary.csv`
- `Final_with_predictions.csv`

---

## 📋 Dataset Format

The input file `Final.csv` must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `Location` | String | Shop location (dropped during preprocessing) |
| `Shop / Owner / Business Name` | String | Shop identifier (dropped during preprocessing) |
| `Plans to Use More Digital Tools (1-5)` | Numeric | **Used to create the target variable** |
| *(other survey columns)* | Numeric (1–5 scale) | Feature inputs for the model |

> All non-numeric columns except `Location` and `Shop / Owner / Business Name` will be coerced to numeric; rows with invalid values are dropped.

---

## 📈 Visualizations Generated

- Correlation heatmap of all features
- Class distribution before and after SMOTE (pie charts)
- Test accuracy bar chart comparing all models
- Confusion matrices (2×2 grid) for all models
- Ensemble weight tuning plot (GridSearchCV — `FinalModel.py` only)
- Feature importance bar chart (Random Forest)
- Digital adoption distribution (pie + bar chart)

---

## 🏷️ Target Variable Definition

```python
Digital Adoption = 1   if "Plans to Use More Digital Tools (1-5)" >= 4
Digital Adoption = 0   otherwise
```

---

## 📝 License

This project is for academic and research purposes.

---

## 👤 Author

**Sumeet Biswal**  
B.Tech Computer Science Engineering, KIIT University  
[LinkedIn](https://www.linkedin.com/in/) • [LeetCode](https://leetcode.com/u/sumeet-biswal/)