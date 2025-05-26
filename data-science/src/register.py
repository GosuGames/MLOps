
import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the trained model")
    args = parser.parse_args()

    # Start the MLflow experiment run
    mlflow.start_run()

    # Load the model
    model = mlflow.sklearn.load_model(args.model)

    print("Registering the best trained model")

    # Logging the model as a registered model with mlflow
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_car_regression_model",
        artifact_path="decision_tree_used_car_regressor"
    )

    # End MLflow run
    mlflow.end_run()
    
if __name__ == "__main__":
    main()
