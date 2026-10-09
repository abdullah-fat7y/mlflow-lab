"""
ML Experiment Tracking with MLflow
House Price Prediction using California Housing dataset
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# Part 1: Prepare the Dataset
# ---------------------------------------------------------
def load_and_split_data(csv_path="california_housing.csv", test_size=0.2, random_state=42):
    """Load California Housing dataset and split into train/validation."""
    df = pd.read_csv(csv_path)

    # The CSV has no 'MedHouseVal' target column (it's the raw features only).
    # If your CSV contains a target column, replace the line below accordingly.
    # For this dataset, we assume the last column or a known target is present.
    # ---------------------------------------------------------
    # IMPORTANT: Adjust this depending on your CSV's target column.
    # The standard California Housing dataset has 'MedHouseVal' as target.
    # If your CSV lacks it, you must add it or use a different target.
    # ---------------------------------------------------------
    if "MedHouseVal" in df.columns:
        target_col = "MedHouseVal"
    elif "target" in df.columns:
        target_col = "target"
    else:
        # Fallback: assume the last column is the target
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------
# Part 2 & 3: Train Multiple Experiments with MLflow Tracking
# ---------------------------------------------------------
def train_and_log(max_depth, learning_rate, X_train, X_val, y_train, y_val, run_name):
    """Train a RandomForest model, evaluate, and log everything to MLflow."""

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # Train model
        # Note: RandomForestRegressor doesn't use learning_rate directly.
        # We use it as a tracked hyperparameter (e.g., could map to a subsample or
        # we could use GradientBoostingRegressor which does use learning_rate).
        # For consistency with the task, we use RandomForest and log learning_rate
        # as a tracked parameter (it won't affect training here).
        # ---------------------------------------------------------
        # To make learning_rate meaningful, we use GradientBoostingRegressor:
        # ---------------------------------------------------------
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"[{run_name}] max_depth={max_depth}, lr={learning_rate} "
              f"-> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return rmse, mae, r2


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Set MLflow experiment name
    mlflow.set_experiment("California_Housing_Experiments")

    # Load and split data
    X_train, X_val, y_train, y_val = load_and_split_data()

    # Define the three experiment configurations
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
            X_train=X_train, X_val=X_val,
            y_train=y_train, y_val=y_val,
            run_name=exp["run_name"]
        )
        results.append({
            "Run": exp["run_name"],
            "Max Depth": exp["max_depth"],
            "Learning Rate": exp["learning_rate"],
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
        })

    # Print comparison table
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Select best model (lowest RMSE)
    best = min(results, key=lambda x: x["RMSE"])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best['Run']} (RMSE = {best['RMSE']})")
    print("=" * 70)


if __name__ == "__main__":
    main()