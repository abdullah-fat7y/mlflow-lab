# ML Experiment Tracking with MLflow

## 1. Task Description

This project demonstrates how to use **MLflow** to track and compare
machine learning experiments for a **House Price Prediction** problem
using the **California Housing dataset**.

Three experiments are trained with different hyperparameters
(`max_depth`, `learning_rate`) using a `GradientBoostingRegressor`.
For each run, MLflow records:

- **Parameters:** `max_depth`, `learning_rate`
- **Metrics:** RMSE, MAE, R²
- **Artifacts:** The trained model

Finally, the best-performing model is selected based on **RMSE**.

<img width="1920" height="1020" alt="Code_tG8RW2i5dE" src="https://github.com/user-attachments/assets/791e176f-2d0a-4026-85c8-f0acb97353ed" />

---

## 2. How to Run the Project

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/mlflow-task.git
cd mlflow-task

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training script
python train.py

# 5. Launch the MLflow UI
mlflow ui
