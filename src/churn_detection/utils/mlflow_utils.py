import mlflow
from churn_detection.utils.config import MLFLOW_TRACKING_URL
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)


def fetch_model(model_name: str):
    """
    Fetch the model in staging from MLflow model registry.

    Args:
        model_name (str): Name of the registered model

    Returns:
        The loaded model from staging
    """
    client = MlflowClient()

    # Get the latest staging model version
    try:
        model_version = client.get_latest_versions(model_name, stages=["staging"])[0]
        model_uri = f"models:/{model_name}/staging"

        # Load the model
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model version {model_version.version} from staging")
        return model

    except Exception as e:
        print(f"Error fetching staging model: {str(e)}")
        return None
