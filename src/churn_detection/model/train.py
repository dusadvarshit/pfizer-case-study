import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import mlflow
from churn_detection.utils.config import (
    MLFLOW_TRACKING_URL,
    binary_cols,
    categorical_cols,
    categorical_cols_mapping,
    ohe_cols,
)
from churn_detection.utils.io import split_train_test

mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)


def build_preprocessor():
    ## Setting Encoders
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
    ordinal_binary = OrdinalEncoder().set_output(transform="pandas")
    ordinal_cat = OrdinalEncoder(categories=categorical_cols_mapping).set_output(transform="pandas")

    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", ohe, ohe_cols),
            ("binary", ordinal_binary, binary_cols),
            ("cat", ordinal_cat, categorical_cols),
        ]
    ).set_output(transform="pandas")

    return preprocessor


def build_pipeline(preprocessor, model) -> Pipeline:
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    return clf


def model_tune(X, y, model_name: str, model, param_grid: dict):
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    with mlflow.start_run(run_name=model_name):
        print(f"---{model_name}----")

        pipeline = build_pipeline(build_preprocessor(), model)

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", refit=True)

        mlflow.set_tag("developer", "Varshit Dusad")
        mlflow.set_tag("model_name", model_name)

        grid_search.fit(X_train, y_train)
        mlflow.log_params(grid_search.best_params_)

        y_predict = grid_search.predict(X_test)
        scores = cross_validate(
            grid_search,
            X_train,
            y_train,
            cv=5,
            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "jaccard",
                "mutual_info_score",
            ],
        )
        scores = {metric: np.mean(value) for metric, value in scores.items()}
        mlflow.log_metrics(scores)

        ## Log Model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

        ## Classification Report
        report = classification_report(y_predict, y_test, output_dict=True)

        mlflow.log_dict(report, artifact_file="classification_report.json")

    mlflow.end_run()


def test():
    print("this worked!")
    return 1
