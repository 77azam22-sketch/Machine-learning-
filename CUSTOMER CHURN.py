# ==========================================
# CUSTOMER CHURN – FULL ML PIPELINE + EDA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("C:/Users/USER PC/Music/customer_churn_dataset-testing-master.csv")

# ------------------------------------------
# 2. DROP CUSTOMER ID
# ------------------------------------------
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# ------------------------------------------
# 3. FEATURE TYPES
# ------------------------------------------
target = "Churn"
cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(target)

# ------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------
print("--- Generating EDA Visualizations ---")

# A. Pie Chart: Churn Distribution
plt.figure(figsize=(6, 6))
churn_counts = df[target].value_counts()
plt.pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', 
        colors=['#66b3ff','#ff9999'], startangle=140, explode=(0, 0.1))
plt.title("Proportion of Churned vs. Retained Customers")
plt.show()

# B. Heatmap: Feature Correlation
plt.figure(figsize=(10, 8))
# Calculating correlation for numerical features
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Numerical Feature Correlation Heatmap")
plt.show()

# C. Bar Charts: Categorical Features vs Churn
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, hue=target, ax=axes[i], palette='viridis')
    axes[i].set_title(f"Churn Relationship with {col}")
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# D. Numerical Feature Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df[df[target] == 1]['Tenure'], color='red', label='Churned', kde=True)
sns.histplot(df[df[target] == 0]['Tenure'], color='blue', label='Not Churned', kde=True)
plt.title("Customer Tenure Distribution")
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(x=target, y='Total Spend', data=df, palette='Set2')
plt.title("Total Spend vs. Churn Status")
plt.show()

# ------------------------------------------
# 6. PREPROCESSING PIPELINE
# ------------------------------------------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# ------------------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------
# 5. OUTLIER REMOVAL (IQR)
# ------------------------------------------
def remove_outliers(data, cols, k=1.5):
    for col in cols:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        data = data[
            (data[col] >= q1 - k * iqr) &
            (data[col] <= q3 + k * iqr)
        ]
    return data

df = remove_outliers(df, num_cols)

X = df.drop(target, axis=1)
y = df[target]

# ------------------------------------------
# 8. MODELS + GRID SEARCH
# ------------------------------------------
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"model__C": [0.01, 0.1, 1, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 10],
            "model__min_samples_split": [2, 5]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    }
}

results = {}

# ------------------------------------------
# 9. TRAIN + EVALUATION
# ------------------------------------------
for name, cfg in models.items():
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", cfg["model"])
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=cfg["params"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "accuracy": acc, "auc": roc_auc, "y_pred": y_pred, "y_prob": y_prob,
        "model": best_model, "best_params": grid.best_params_
    }

    print("\n" + "="*50)
    print(f"MODEL: {name}")
    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ------------------------------------------
# 10. ROC CURVE COMPARISON
# ------------------------------------------
plt.figure(figsize=(8,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ------------------------------------------
# 11. FEATURE IMPORTANCE
# ------------------------------------------
for model_name in ["Random Forest", "Gradient Boosting"]:
    model = results[model_name]["model"].named_steps["model"]
    feature_names = results[model_name]["model"].named_steps["prep"].get_feature_names_out()
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="magma")
    plt.title(f"Top 10 Feature Importances – {model_name}")
    plt.show()

   # 1. Rank features by their correlation with Churn
correlations = df.corr()[target].sort_values(ascending=False)
print("--- Correlation with Churn ---")
print(correlations)
