# =============================================================================
# DIGITAL ADOPTION PREDICTION MODEL
# Dataset: Final.csv
# =============================================================================

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

print("Libraries loaded successfully.")


# =============================================================================
# STEP 2: LOAD DATASET
# =============================================================================
df = pd.read_csv("Final.csv")

print("Dataset loaded successfully")
print(df.head().to_string())


# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================

# Remove non-numeric identifier columns
columns_to_drop = ["Location", "Shop / Owner / Business Name"]

df_clean = df.drop(columns=columns_to_drop)

print("\nColumns after cleaning:")
print(df_clean.columns)

print("\nData types:")
print(df.dtypes)


# =============================================================================
# STEP 4: CREATE TARGET VARIABLE
# =============================================================================

# Convert column to numeric
df_clean["Plans to Use More Digital Tools (1-5)"] = pd.to_numeric(
    df_clean["Plans to Use More Digital Tools (1-5)"],
    errors="coerce"
)

# Create target variable
df_clean["Digital Adoption"] = df_clean["Plans to Use More Digital Tools (1-5)"].apply(
    lambda x: 1 if x >= 4 else 0
)

print("\nTarget variable created.")
print("\nTarget distribution:")
print(df_clean["Digital Adoption"].value_counts())

df_clean.drop(columns=["Plans to Use More Digital Tools (1-5)"], inplace=True)


# =============================================================================
# STEP 5: CONVERT ALL FEATURE COLUMNS TO NUMERIC
# =============================================================================

for column in df_clean.columns:
    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')

df_clean = df_clean.dropna()
df_clean = df_clean.astype(int)

print("\nAll columns converted to numeric successfully.")
print("New shape:", df_clean.shape)
print(df_clean.to_string())


# =============================================================================
# STEP 5.5: CORRELATION HEATMAP
# =============================================================================
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean.corr(), annot=True, fmt='.1f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 5.6: FEATURE SELECTION BY CORRELATION
# =============================================================================
corr = df_clean.corr()

# Get correlation with target
target_corr = corr["Digital Adoption"]

# Filter features with abs correlation >= 0.2
selected_features = target_corr[abs(target_corr) >= 0.2].index

# Remove target itself
selected_features = selected_features.drop("Digital Adoption")

X = df_clean[selected_features]
y = df_clean["Digital Adoption"]

print("Selected Features:\n", selected_features)
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# =============================================================================
# STEP 6: TRAIN TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split done.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =============================================================================
# STEP 6.5: CLASS DISTRIBUTION BEFORE SMOTE
# =============================================================================
y_train.value_counts().plot.pie(
    autopct='%1.1f%%',
    figsize=(5, 5),
    startangle=90
)
plt.title("Class Distribution Before SMOTE (Training Data)")
plt.ylabel('')
plt.tight_layout()
plt.show()

print("Class distribution before SMOTE:")
print(y_train.value_counts())


# =============================================================================
# STEP 6.6: APPLY SMOTE
# =============================================================================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train_sm.value_counts())

y_train_sm.value_counts().plot.pie(
    autopct='%1.1f%%',
    figsize=(5, 5),
    startangle=90
)
plt.title("Class Distribution After SMOTE")
plt.ylabel('')
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 7: TRAIN MODELS (WITH SMOTE DATA)
# =============================================================================

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_sm, y_train_sm)
print("\nRandom Forest trained.")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_sm, y_train_sm)
print("Decision Tree trained.")

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train_sm, y_train_sm)
print("Naive Bayes trained.")

# Ensemble Model (Soft Voting)
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('dt', dt),
        ('gnb', gnb)
    ],
    voting='soft'
)
ensemble.fit(X_train_sm, y_train_sm)
print("Ensemble model trained.")


# =============================================================================
# STEP 8: MODEL EVALUATION (TEST SET)
# =============================================================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

models = {
    "Random Forest": rf,
    "Decision Tree": dt,
    "Naive Bayes": gnb,
    "Ensemble": ensemble
}

test_accuracies = {}
predictions = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    test_accuracies[name] = acc

    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# =============================================================================
# STEP 8.5: BAR CHART - ACCURACY COMPARISON
# =============================================================================
plt.figure()
bars = plt.bar(test_accuracies.keys(), test_accuracies.values())
plt.title("Test Accuracy Comparison of Models")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.ylim(0, 1)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.3f}',
        ha='center',
        va='bottom'
    )
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 9: SELECT BEST MODEL
# =============================================================================
best_model_name = max(test_accuracies, key=test_accuracies.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", test_accuracies[best_model_name])


# =============================================================================
# STEP 10: FINAL PREDICTION, SUMMARY, AND OUTPUT EXPORT
# =============================================================================
print("\n" + "="*80)
print("DIGITAL ADOPTION PREDICTION ANALYSIS - FINAL RESULTS")
print("="*80)

# 1. Generate Predictions
full_predictions = best_model.predict(X)

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(X)[:, 1]
else:
    probabilities = [0] * len(full_predictions)

# 2. Create Final Output DataFrame
df_final = df.loc[df_clean.index].copy()
df_final["Predicted Adoption"] = full_predictions
df_final["Adoption Probability (%)"] = probabilities * 100

# 3. Add Interpretation Column
df_final["Adoption Category"] = df_final["Predicted Adoption"].map({
    1: "Likely Adopter",
    0: "Unlikely Adopter"
})

# 4. Calculate Summary Statistics
total_shops = len(df_final)
total_adopters = (df_final["Predicted Adoption"] == 1).sum()
total_non_adopters = (df_final["Predicted Adoption"] == 0).sum()
adoption_rate = (total_adopters / total_shops) * 100

# 5. Display Summary
print("\nSUMMARY STATISTICS:")
print("-"*50)
print(f"Total Shops Analyzed        : {total_shops}")
print(f"Likely Digital Adopters    : {total_adopters}")
print(f"Unlikely Digital Adopters  : {total_non_adopters}")
print(f"Adoption Rate              : {adoption_rate:.2f}%")

print("\nMODEL INFORMATION:")
print("-"*50)
print(f"Best Model Used            : {best_model_name}")

model_accuracy = accuracy_score(y, best_model.predict(X))
print(f"Overall Model Accuracy     : {model_accuracy*100:.2f}%")

# 6. Show Sample Predictions
print("\nSAMPLE PREDICTIONS:")
print("-"*50)
print(df_final.head(10).to_string())

# 7. Add Project Summary Columns
df_final["Total Shops Analyzed"] = total_shops
df_final["Total Likely Adopters"] = total_adopters
df_final["Total Unlikely Adopters"] = total_non_adopters
df_final["Overall Adoption Rate (%)"] = adoption_rate
df_final["Model Used"] = best_model_name
df_final["Model Accuracy (%)"] = model_accuracy * 100

# 8. Save Final Output File
output_file = "Final_with_predictions_and_summary.csv"
df_final.to_csv(output_file, index=False)

print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print(f"\nOutput file saved as: {output_file}")
print("\nThis file contains:")
print("• Original shop data")
print("• Predicted digital adoption")
print("• Adoption probability")
print("• Adoption category")
print("• Model accuracy")
print("• Full project summary")
print("\nAnalysis completed successfully.")


# =============================================================================
# STEP 10.5: ADOPTION DISTRIBUTION CHARTS
# =============================================================================
df_final["Adoption Category"].value_counts().plot.pie(
    autopct='%1.1f%%',
    figsize=(6, 6),
    startangle=90
)
plt.title("Digital Adoption Distribution")
plt.ylabel('')
plt.tight_layout()
plt.show()

df_final["Adoption Category"].value_counts().plot(kind='bar')
plt.title("Digital Adoption Count")
plt.xlabel("Category")
plt.ylabel("Number of Shops")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 11: CONFUSION MATRICES (TEST DATA)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )
    acc = test_accuracies[name]
    ax.set_title(f"{name} (Test)\nAccuracy: {acc:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices (Test Set)", fontsize=16)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 12: GRIDSEARCHCV - OPTIMIZE ENSEMBLE WEIGHTS
# =============================================================================
param_grid = {
    'weights': [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 2, 2)],
    'voting': ['soft']
}

grid_ensemble = GridSearchCV(
    estimator=ensemble,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_ensemble.fit(X_train_sm, y_train_sm)

print("Best Ensemble Params:", grid_ensemble.best_params_)
print("Best CV Accuracy:", grid_ensemble.best_score_)

# Update ensemble with best estimator
ensemble = grid_ensemble.best_estimator_

# Plot CV Results
results = grid_ensemble.cv_results_
weights = results['param_weights']
scores = results['mean_test_score']
weights_str = [str(w) for w in weights]

plt.figure(figsize=(8, 5))
plt.plot(weights_str, scores, marker='o')
plt.xlabel("Weights (RF, DT, NB)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Accuracy vs Ensemble Weights")
plt.xticks(rotation=45)

for x, score in zip(weights_str, scores):
    plt.text(x, score, f"{score:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


# =============================================================================
# STEP 13: SAVE RESULTS
# =============================================================================
output_file = "Final_with_predictions.csv"
df_final.to_csv(output_file, index=False)
print("\nResults saved successfully to:", output_file)


# =============================================================================
# STEP 14: FEATURE IMPORTANCE (ALWAYS FROM RANDOM FOREST)
# =============================================================================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance.to_string())

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 15: FINAL SUMMARY
# =============================================================================
print("="*60)
print("FINAL SUMMARY")
print("="*60)

total_shops = len(df_final)
total_adopters = (df_final["Predicted Adoption"] == 1).sum()
total_non_adopters = (df_final["Predicted Adoption"] == 0).sum()

print(f"Total Shops                : {total_shops}")
print(f"Predicted Adopters         : {total_adopters}")
print(f"Predicted Non-Adopters     : {total_non_adopters}")
print(f"Adoption Rate (%)          : {(total_adopters / total_shops) * 100:.2f}")

print("\nBest Model Used            :", best_model_name)
print(f"Test Accuracy              : {test_accuracies[best_model_name]*100:.2f}%")

print("\nAnalysis Complete.")
