import pandas as pd
from flask import Flask, jsonify, render_template, request

from churn_detection.utils.mlflow_utils import fetch_model

app = Flask(__name__)

# Load the model from MLflow

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
# logged_model = 'runs:/5009d9dbd0d243e8ab66b5b4db2f0530/model'
# model = mlflow.sklearn.load_model(logged_model)


@app.route("/")
def home():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert JSON to DataFrame
        print(data)
        df = pd.DataFrame([data])

        # Make predictions
        model = fetch_model()
        predictions = model.predict(df)

        # Return predictions as JSON
        predictions = predictions.tolist()[0]

        if predictions == 1:
            return jsonify({"predictions": "The customer will CHURN!"})

        return jsonify({"predictions": "The customer will NOT Churn!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
