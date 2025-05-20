import pandas as pd
from flask import Flask, jsonify, render_template, request

from churn_detection.utils.logger import CustomLogger
from churn_detection.utils.mlflow_utils import fetch_model

logger = CustomLogger("App").get_logger()

app = Flask(__name__)
model = fetch_model("staging")


@app.route("/")
def home():
    return render_template("form.html")


@app.route("/get", methods=["GET"])
def get():
    logger.info("Get page")
    model = fetch_model("staging")
    logger.info(f"Model Loaded! Type {type(model)}")
    return "Connection Succeeded"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert JSON to DataFrame
        logger.info(data)
        df = pd.DataFrame([data])

        # Make predictions
        logger.info("Before prediction")
        predictions = model.predict(df)
        logger.info("After prediction")

        # Return predictions as JSON
        predictions = predictions.tolist()[0]

        if predictions == 1:
            return jsonify({"predictions": "The customer will CHURN!"})

        return jsonify({"predictions": "The customer will NOT Churn!"})

    except Exception as e:
        logger.info(f"ERROR during prediction! {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
