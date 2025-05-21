# 🧠 Customer Churn Detection - End-to-End ML Pipeline

A modular, production-ready machine learning pipeline for detecting customer churn. This repository combines best practices in machine learning engineering — from model training and evaluation to deployment with a Flask app, tracking with MLflow, containerization with Docker, and rigorous code quality with pre-commit hooks and unit testing.

---

## 📚 Table of Contents

- [🔧 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🏋️‍♂️ Training the Model](#️-training-the-model)
- [🧪 Running Tests](#-running-tests)
- [🌐 Serving via Web App](#-serving-via-web-app)
- [📦 Docker Deployment](#-docker-deployment)
- [📊 MLflow Integration](#-mlflow-integration)
- [💡 API Usage Example](#-api-usage-example)
- [🧰 Development Practices](#-development-practices)
- [📁 Notebooks & Data](#-notebooks--data)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🔧 Features

- ✅ End-to-end churn prediction workflow
- ✅ Clean, modular Python codebase
- ✅ Flask web app for local and containerized deployment
- ✅ MLflow for experiment tracking
- ✅ Docker support for reproducible environments
- ✅ Pre-commit hooks (Black, Flake8, isort, etc.)
- ✅ Pytest unit testing
- ✅ Logging and configuration management

---

## 📁 Project Structure

```text
src/
└── churn_detection/
    ├── main.py                   # Entry point (train, evaluate, predict)
    ├── deployment/
    │   └── deployment/
    │       ├── app.py            # Flask app server
    │       ├── static/
    │       │   └── style.css     # Styling for HTML frontend
    │       └── templates/
    │           └── form.html     # Input form UI
    ├── model/
    │   └── model/
    │       ├── train.py          # Model training logic
    │       ├── eval.py           # Evaluation metrics
    │       ├── model_loader.py   # Save/load model functions
    └── utils/
        └── utils/
            ├── config.py        # Configuration and hyperparameters
            ├── io.py            # Data loading and I/O utilities
            ├── logger.py        # Logging setup
            ├── mlflow_utils.py  # MLflow integration
tests/
└── tests/
    └── __init__.py               # Unit test definitions
Dockerfile                        # Docker build instructions

```

## 🔧 Features

- ✅ End-to-end churn prediction workflow
- ✅ Clean, modular Python codebase
- ✅ Flask web app for local and containerized deployment
- ✅ MLflow for experiment tracking
- ✅ Docker support for reproducible environments
- ✅ Pre-commit hooks (Black, Flake8, isort, etc.)
- ✅ Pytest unit testing
- ✅ Logging and configuration management

---

# 🚀 Getting Started

## 1. Clone the Repository
```
git clone https://github.com/yourusername/churn-detection.git
cd churn-detection/src
```


# 🧪 Running Tests
```
pytest tests/
```
