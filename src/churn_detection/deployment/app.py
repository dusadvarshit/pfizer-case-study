import pandas as pd
from flask import Flask, jsonify, render_template, request

from churn_detection.utils.logger import CustomLogger
from churn_detection.utils.mlflow_utils import fetch_model

logger = CustomLogger("App").get_logger()

# if os.environ["ENV"] == "DEV":
#     model = fetch_model("staging")
# elif if os.environ["ENV"] == "PROD":
#     model = fetch_model("production")

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


@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert JSON to DataFrame
        logger.info(data)
        df = pd.DataFrame(data)

        # Make predictions
        logger.info("Before prediction")
        predictions = model.predict(df)
        logger.info("After prediction")

        # Return predictions as JSON
        predictions = predictions.tolist()
        predictions = ["Yes" if i == 1 else "No" for i in predictions]
        return_data = []
        for idx, _dict in enumerate(data):
            _dict["churn_prediction"] = predictions[idx]
            return_data.append(_dict)

        return jsonify(return_data)

    except Exception as e:
        logger.info(f"ERROR during prediction! {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
