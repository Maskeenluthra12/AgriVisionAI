#Crop Classification Using ML Algorithms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
# 1. LOAD DATASET
df = pd.read_csv("crop_production.csv")

print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())

# 2. DATA PREPROCESSING
df.columns = df.columns.str.strip()

categorical_cols = ["State_Name", "District_Name", "Season"]
numerical_cols = ["Crop_Year", "Area"]

# Remove missing values in required columns
df = df.dropna(
    subset=categorical_cols + numerical_cols + ["Crop"]
)

# Remove duplicate records
df = df.drop_duplicates()

# Remove crop classes with fewer than 5 records
crop_counts = df["Crop"].value_counts()
valid_crops = crop_counts[crop_counts >= 5].index
df = df[df["Crop"].isin(valid_crops)]

# Input features and target
X = df[categorical_cols + numerical_cols]
y = df["Crop"]

print("\nNumber of Records:", len(df))
print("Number of Crop Classes:", y.nunique())

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Limit training samples per crop to make
# cross-validation computationally manageable
train_data = X_train.copy()
train_data["Crop"] = y_train

train_data = (
    train_data.groupby("Crop", group_keys=False)
    .sample(n=200, random_state=42)
)

y_sample = train_data["Crop"]
X_sample = train_data.drop(columns=["Crop"])

# Common test sample for fair model comparison
test_data = X_test.copy()
test_data["Crop"] = y_test

test_data = (
    test_data.groupby("Crop", group_keys=False)
    .sample(n=50, random_state=42)
)

y_test_sample = test_data["Crop"]
X_test_sample = test_data.drop(columns=["Crop"])

# 4. FEATURE PREPROCESSING
def create_preprocessor(scaler=StandardScaler()):
    return ColumnTransformer([
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),

        ("numerical",
         Pipeline([
             ("scaler", scaler)
         ]),
         numerical_cols)
    ])

# Gradient Boosting uses ordinal encoding to keep
# its input compact and computationally manageable.
from sklearn.preprocessing import OrdinalEncoder

gb_preprocessor = ColumnTransformer([
    ("categorical",
     OrdinalEncoder(
         handle_unknown="use_encoded_value",
         unknown_value=-1
     ),
     categorical_cols),

    ("numerical", StandardScaler(), numerical_cols)
])

# 5. DEFINE SIX CLASSIFIERS
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", create_preprocessor(
            StandardScaler(with_mean=False))),
        ("classifier", LogisticRegression(
            max_iter=250,
            solver="saga",
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            max_depth=30,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("preprocessor", create_preprocessor(MinMaxScaler())),
        ("classifier", MultinomialNB(alpha=0.5))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", gb_preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
}

# 6. 5-FOLD CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\n========== CROSS VALIDATION ==========")

for name, model in models.items():

    print("\nTraining:", name)

    scores = cross_val_score(
        model,
        X_sample,
        y_sample,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    print("Fold Scores:", scores)
    print("Average CV Accuracy:", scores.mean())

    results.append({
        "Model": name,
        "CV Accuracy": scores.mean(),
        "CV Std": scores.std()
    })

# 7. MODEL TRAINING AND TESTING
test_results = []
trained_models = {}

print("\n========== MODEL EVALUATION ==========")

for name, model in models.items():

    print("\nModel:", name)

    model.fit(X_sample, y_sample)

    predictions = model.predict(X_test_sample)

    accuracy = accuracy_score(
        y_test_sample, predictions
    )

    precision = precision_score(
        y_test_sample, predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test_sample, predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test_sample, predictions,
        average="macro",
        zero_division=0
    )

    test_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

# 8. COMPARISON TABLE
comparison = pd.DataFrame(test_results)

cv_results = pd.DataFrame(results)

final_results = comparison.merge(
    cv_results, on="Model"
)

final_results = final_results.sort_values(
    by="F1 Score", ascending=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(final_results.to_string(index=False))

final_results.to_csv(
    "classifier_comparison_results.csv",
    index=False
)

# 9. IDENTIFY BEST MODEL
best_model_name = final_results.iloc[0]["Model"]

print("\nBest Performing Model:", best_model_name)

best_model = trained_models[best_model_name]

# 10. CONFUSION MATRIX
best_predictions = best_model.predict(X_test_sample)

cm = confusion_matrix(
    y_test_sample,
    best_predictions,
    labels=best_model.classes_
)

# Display top 15 crop categories
top_crops = y_test_sample.value_counts().head(15).index
indices = [
    list(best_model.classes_).index(crop)
    for crop in top_crops
]

cm_top = cm[np.ix_(indices, indices)]

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm_top,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=top_crops,
    yticklabels=top_crops
)

plt.title("Best Model - Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 11. MODEL COMPARISON GRAPH
plot_data = final_results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

plot_data.plot(
    kind="bar",
    figsize=(13, 6)
)

plt.title("Comparison of Six ML Classifiers")
plt.ylabel("Performance Score")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=25)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# 12. CLASSIFICATION REPORT
print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test_sample,
    best_predictions,
    zero_division=0
))

print("\nAll models evaluated successfully!")

