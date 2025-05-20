import sqlite3

import pandas as pd

# Open the database
conn = sqlite3.connect("mlflow.db")

# Read the experiments table
experiments = pd.read_sql_query("SELECT * FROM experiments", conn)
runs = pd.read_sql_query("SELECT * FROM runs", conn)
model_versions = pd.read_sql_query("SELECT * FROM model_versions", conn)

# Update artifact_location to new location inside Docker
new_artifact_location = "file:/home/pfizer-case-study/mlflow/artifacts"

experiments["artifact_location"] = experiments["artifact_location"].apply(lambda x: x.replace("file:C:/Users/dusad/Documents/Projects/pfizer-case-study/src/churn_detection/../../mlflow/artifacts", new_artifact_location))
experiments.to_sql("experiments", conn, if_exists="replace", index=False)

runs["artifact_uri"] = runs["artifact_uri"].apply(lambda x: x.replace("file:C:/Users/dusad/Documents/Projects/pfizer-case-study/src/churn_detection/../../mlflow/artifacts", new_artifact_location))
runs.to_sql("runs", conn, if_exists="replace", index=False)

model_versions["source"] = model_versions["source"].apply(lambda x: x.replace("file:C:/Users/dusad/Documents/Projects/pfizer-case-study/src/churn_detection/../../mlflow/artifacts", new_artifact_location))
model_versions["storage_location"] = model_versions["storage_location"].apply(lambda x: x.replace("file:C:/Users/dusad/Documents/Projects/pfizer-case-study/src/churn_detection/../../mlflow/artifacts", new_artifact_location))
model_versions.to_sql("model_versions", conn, if_exists="replace", index=False)

conn.commit()
conn.close()
