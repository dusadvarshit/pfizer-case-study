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
# Change Permission
RUN chmod -R 777 /usr/local

WORKDIR /home/pfizer-case-study

# Install Poetry
RUN python3 -m pip install --upgrade poetry
RUN poetry config virtualenvs.create false

# Copy files
COPY pyproject.toml poetry.lock ./


# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi && \
    poetry cache clear --all pypi -n && \
    rm -rf ~/.cache/pypoetry
