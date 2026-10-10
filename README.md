# MLflow Experiment Tracking — House Price Prediction

## 1. Project Overview

This project demonstrates how to use **MLflow** to track, compare, and select machine learning experiments for a house price prediction task using the California Housing dataset.

A regression model was trained using three different hyperparameter configurations. MLflow was used to record the model parameters, validation metrics (RMSE, MAE, and R²), and trained model artifacts.

The objective is to identify the best-performing model based on the **Root Mean Squared Error (RMSE)**.

## 2. Technologies Used

- Python
- Pandas
- Scikit-learn
- MLflow
- California Housing dataset

## 3. Project Structure

```text
mlflow-task/
├── california_housing.csv
├── train.py
├── requirements.txt
└── README.md
```

## 4. Installation and Execution

### Prerequisites

- Python 3.10 or later
- pip

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mlflow-task
```

Alternatively, download the repository and open the project directory in your terminal.

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can run the Python executable inside the virtual environment directly instead.

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Run the experiments

```powershell
python train.py
```

The script trains three model configurations, evaluates their performance on the validation dataset, and logs the experiment results to MLflow.

### Step 5: Open the MLflow UI

If the training script uses the default local MLflow tracking store, start the interface with:

```powershell
mlflow ui
```

Open the following address in your browser:

http://127.0.0.1:5000

Navigate to the `California_Housing_Experiments` experiment to compare the training runs.

> If your script configures a different tracking URI, use the same tracking store when launching or accessing the MLflow UI.

## 5. Experiment Configurations

Three experiments were conducted using different values for `max_depth` and `learning_rate`.

| Experiment | Max Depth | Learning Rate |
|---|---:|---:|
| Run 1 | 3 | 0.10 |
| Run 2 | 5 | 0.05 |
| Run 3 | 7 | 0.01 |

The same dataset split and evaluation metrics were used to make the results comparable.

## 6. Experiment Results

The following table summarizes the validation results produced by the training script.

| Run | Max Depth | Learning Rate | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Run 1 | 3 | 0.10 | 0.4956 | 0.3208 | 0.9382 |
| Run 2 | 5 | 0.05 | **0.4632** | **0.2938** | **0.9460** |
| Run 3 | 7 | 0.01 | 0.8602 | 0.7290 | 0.8139 |

### MLflow Experiment Results Screenshot

Add the screenshot showing the three training runs and their metric comparisons below.

**Screenshot 1 — Training Runs and Metrics**

<img width="1910" height="915" alt="msedge_mwvl1kREcp" src="https://github.com/user-attachments/assets/0282820d-9421-446a-b303-8a3e7521b05c" />


### MLflow Model Registry Screenshot

Add the screenshot showing the registered model and its version below.

**Screenshot 2 — Registered Model: House_Price_Predictor**

<img width="1910" height="915" alt="msedge_wUiF2cEu7L" src="https://github.com/user-attachments/assets/f1cf7a9a-36ac-41c5-b658-de71dc4f18f0" />


## 7. Best Model Selection

**Selected Model: Run 2**

- **Max Depth:** 5
- **Learning Rate:** 0.05
- **RMSE:** 0.4632
- **MAE:** 0.2938
- **R²:** 0.9460

Run 2 was selected because it achieved the **lowest RMSE (0.4632)** among the three experiments. Since RMSE is the primary selection metric for this task, the lowest value indicates the best prediction performance among the tested configurations.

Run 2 also achieved the lowest MAE and the highest R², providing additional evidence that it performed best on the validation dataset.

Run 1 performed reasonably well, but its errors were slightly higher. Run 3 produced substantially higher errors and a lower R² score, making it the weakest configuration of the three.

## 8. Conclusion

This project demonstrates the MLflow experiment-tracking workflow:

**Train → Track → Compare → Select**

By logging model parameters, validation metrics, and trained model artifacts, MLflow makes it easier to compare experiments and identify the best-performing configuration.

Based on the validation results, **Run 2 (`max_depth=5`, `learning_rate=0.05`) is the selected model**.

## 9. Author

**Abdullah Fathy**

GitHub: [abdullah-fat7y](https://github.com/abdullah-fat7y)
