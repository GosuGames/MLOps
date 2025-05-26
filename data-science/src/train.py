import mlflow
import argparse
import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def select_first_file(path):
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument('--criterion', type=str, default='squared_error',
                        help='The function to measure the quality of a split (e.g., squared_error)')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree.')
    parser.add_argument("--model_output", type=str, help="Path of output model")
    args = parser.parse_args()

    mlflow.start_run()  # Start MLflow run

    # Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Features and target
    X_train = train_df.drop("price", axis=1)
    y_train = train_df["price"].astype(float)
    X_test = test_df.drop("price", axis=1)
    y_test = test_df["price"].astype(float)

    # One-hot encode categorical features (e.g., Segment)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Train Decision Tree Regressor
    tree_model = DecisionTreeRegressor(criterion=args.criterion, max_depth=args.max_depth)
    tree_model.fit(X_train, y_train)
    predictions = tree_model.predict(X_test)

    # Evaluate with Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print(f'MSE of Decision Tree regressor on test set: {mse:.2f}')
    mlflow.log_metric("MSE", float(mse))

    # Output the model
    mlflow.sklearn.save_model(tree_model, args.model_output)
    mlflow.end_run()

if __name__ == "__main__":
    main()
