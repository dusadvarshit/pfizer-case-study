FROM dusadvarshit12/mlapp-base-env:latest AS system
SHELL ["/bin/bash", "-c"]
USER root

WORKDIR /home/pfizer-case-study

# Install Poetry
RUN python3 -m pip install --upgrade poetry
RUN poetry config virtualenvs.create false

# Copy files
COPY . .

#set permissions
RUN chmod -R 777 /home/pfizer-case-study
WORKDIR /home/pfizer-case-study/src

# entrypoint
EXPOSE 5000
CMD ["python3", "-m", "churn_detection.deployment.app"]
