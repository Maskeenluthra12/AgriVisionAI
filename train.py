
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/crop_production.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--experiment_name", default="AgriVisionAI")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    df = pd.read_csv(data_path)

    features = [
        "State_Name", "District_Name", "Crop_Year",
        "Season", "Crop", "Area"
    ]
    target = "Production"

    required = features + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df[required].dropna(subset=[target]).drop_duplicates()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce")
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df = df.dropna(subset=[target, "Crop_Year", "Area"])

    X = df[features]
    y = df[target]

    categorical = ["State_Name", "District_Name", "Season", "Crop"]
    numerical = ["Crop_Year", "Area"]

    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numerical", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numerical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    models = {
        "DecisionTree": DecisionTreeRegressor(
            max_depth=args.max_depth,
            random_state=args.random_state
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
    }

    mlflow.set_experiment(args.experiment_name)

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            r2 = r2_score(y_test, predictions)

            mlflow.log_param("model", name)
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            mlflow.sklearn.log_model(
    pipeline,
    name="model",
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree"
    ]
)

            results.append({
                "model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(
                f"{name}: MAE={mae:.3f}, "
                f"RMSE={rmse:.3f}, R2={r2:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(
        by="MAE", ascending=True
    )

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]

    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])
    best_pipeline.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_pipeline, "models/model.joblib")

    metrics = {
        "best_model": best_name,
        "rows_used": int(len(df)),
        "features": features,
        "target": target,
        "results": results
    }

    with open("models/training_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    results_df.to_csv("models/model_comparison.csv", index=False)

    print("\nBest model:", best_name)
    print("Saved model to models/model.joblib")
    print("Saved metrics to models/training_metrics.json")


if __name__ == "__main__":
    main()
