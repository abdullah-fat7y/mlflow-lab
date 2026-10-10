"""
ML Experiment Tracking with MLflow
House Price Prediction using the California Housing dataset
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Types skops must be allowed to load when saving the model.
# Safe here because we trained this model ourselves.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


# ---------------------------------------------------------
# Part 1: Prepare the dataset
# ---------------------------------------------------------
def load_and_split_data(csv_path="california_housing.csv", test_size=0.2, random_state=42):
    """Load the dataset and split it into train/validation sets."""
    df = pd.read_csv(csv_path)

    # Target column: 'MedHouseVal' or 'target' if present, otherwise the last column.
    if "MedHouseVal" in df.columns:
        target_col = "MedHouseVal"
    elif "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------
# Parts 2 & 3: Train experiments and log them with MLflow
# ---------------------------------------------------------
def train_and_log(max_depth, learning_rate, X_train, X_val, y_train, y_val, run_name):
    """Train a GradientBoostingRegressor, evaluate it, and log everything to MLflow."""
    with mlflow.start_run(run_name=run_name):
        # Parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # Train (GradientBoosting actually uses learning_rate, unlike RandomForest)
        model = GradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Model artifact
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=TRUSTED_TYPES,
        )

        print(
            f"[{run_name}] max_depth={max_depth}, lr={learning_rate} "
            f"-> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )

        return rmse, mae, r2


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    mlflow.set_experiment("California_Housing_Experiments")

    X_train, X_val, y_train, y_val = load_and_split_data()

    experiments = [
        {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
        {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
        {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
    ]

    results = []
    for exp in experiments:
        rmse, mae, r2 = train_and_log(
            max_depth=exp["max_depth"],
            learning_rate=exp["learning_rate"],
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            run_name=exp["run_name"],
        )
        results.append(
            {
                "Run": exp["run_name"],
                "Max Depth": exp["max_depth"],
                "Learning Rate": exp["learning_rate"],
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
            }
        )

    # Comparison table
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print(pd.DataFrame(results).to_string(index=False))

    # Best model (lowest RMSE)
    best = min(results, key=lambda r: r["RMSE"])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best['Run']} (RMSE = {best['RMSE']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
