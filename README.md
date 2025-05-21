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

``
---

# 🚀 Getting Started

## 1. Clone the Repository
```
git clone https://github.com/yourusername/churn-detection.git
cd churn-detection/src
```


# 🧪 Running Tests
Use pytest to run all unit tests:
```
pytest tests/
```

For test coverage:
```
pytest --cov=churn_detection tests/
```

# 🌐 Serving via Web App
You can start a local Flask web server:
```
cd src/
python -m churn_detection.deployment.deployment.app
```
Visit your browser at: http://localhost:5000

Here, you can input customer details in a form and get real-time churn predictions.

# 📦 Docker Deployment

1. Build Docker Image - App
Make sure you are in the root directory (where the Dockerfile is located):
```
docker build -f base-env.dockerfile -t mlapp-base-env
docker build -t churn-detection-app .
```

2. Build Docker Image - Mlflow
Make sure you are in the mlflow directory (where the Dockerfile for mlflow is located):
```
docker build -t mlflow .
```

3. Deploy using a docker network
```
docker network create my-app-network

docker run -d --name mlflow --network my-app-network -p 8000:8000 mlflow

docker run -d --name mlapp --network my-app-network -p 5000:5000 mlapp
```

# 🧰 Development Practices
This repository follows strict development standards:

Git version control

Pre-commit hooks for linting, formatting, and secrets

Docker for repeatable environments

MLflow for reproducible ML experiments

Logging to track events

Unit tests for all major components.

# 📁 Notebooks & Data
Jupyter notebooks for data exploration and EDA are in the /notebooks directory.

Training and test datasets should be placed in /data.

Models are saved to /models.

None of these folders are put into .gitignore purposefully to make this entire notebook reproducible.

# 🤝 Contributing

Contributions are welcome!

Fork the repo

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Ensure you follow the code standards and include relevant unit tests.

# 📄 License
Distributed under the MIT License. See LICENSE for more information.

# 🙌 Acknowledgments
This project integrates tools and practices from:

Scikit-learn

Flask

MLflow

Docker

Pandas / NumPy

Pytest

Pre-commit

GitHub Actions (optional CI/CD integration)

**This file is completely self-contained and fully detailed. You can copy this directly into your project as `README.md` without needing any additional documents. Let me know if you'd like me to generate a badge section or add sample config templates in-code!**
