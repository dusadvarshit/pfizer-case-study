from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MLFLOW_TRACKING_URL = "http://localhost:8000"

MLFLOW_ROOT_FOLDER = "../../../mlflow"

MLFLOW_ARTIFACTS = "../../../mlflow/artifacts"

HYPERPARAMS_DICT = binary_classification_models_param_grid = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [10, 100, 1000],
        },
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5],
        },
    },
    "AdaBoostClassifier": {
        "model": AdaBoostClassifier(),
        "param_grid": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1, 1],
        },
    },
    "BaggingClassifier": {
        "model": BaggingClassifier(),
        "param_grid": {
            "model__n_estimators": [10, 50, 100],
            "model__max_samples": [0.5, 1.0],
            "model__max_features": [0.5, 1.0],
        },
    },
    "SVC": {
        "model": SVC(),
        "param_grid": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"],
        },
    },
    "DecisionTreeClassifier": {
        "model": DecisionTreeClassifier(),
        "param_grid": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 10],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "param_grid": {
            # No major hyperparameters to tune, included for completeness
        },
    },
    "XGBClassifier": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        },
    },
    "LGBMClassifier": {
        "model": LGBMClassifier(),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5, 7],
        },
    },
}

ohe_cols = ["PaymentMethod"]
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
categorical_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
]
categorical_cols_mapping = [
    ["No phone service", "No", "Yes"],
    ["No", "DSL", "Fiber optic"],
    *([["No internet service", "No", "Yes"]] * 6),
    ["Month-to-month", "One year", "Two year"],
]
