FROM python:3.10-slim AS system
SHELL ["/bin/bash", "-c"]
USER root

#system dependencies
RUN apt-get update && apt-get install -y \
    build-essential python3-dev zlib1g-dev libffi-dev libpq-dev libssl-dev libexpat1-dev \
    libz-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## utilities and tool. install and setup
FROM system AS utilities

SHELL ["/bin/bash", "-c"]
USER root
RUN chmod 777 /home

# Upgrade pip
RUN python3 -m pip install -U pip

# Install mlflow
RUN pip install --no-cache-dir --no-compile mlflow pandas
RUN pip cache purge

# Change Permission
RUN chmod -R 777 /usr/local

WORKDIR /home/pfizer-case-study/mlflow

# Copy files
COPY . .
RUN python artifact_location_fix.py

#set permissions
RUN chmod -R 777 /home/pfizer-case-study/mlflow

# entrypoint
EXPOSE 8000
CMD ["mlflow", "server", "--backend-store-uri",  "sqlite:///mlflow.db", "--default-artifact-root", "artifacts", "--host", "0.0.0.0", "--port", "8000"]
