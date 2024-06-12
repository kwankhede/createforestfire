from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


## import ridge and stander scaler pickle
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")  # Home page
def index():
    return render_template("index.html")  # it will file the index.html in templates


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Extracting form data and converting to float
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Create a DataFrame with the input data
        input_data = pd.DataFrame(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
            columns=[
                "Temperature",
                "RH",
                "Ws",
                "Rain",
                "FFMC",
                "DMC",
                "ISI",
                "Classes",
                "Region",
            ],
        )

        # Standardize the input data
        scaled_data = standard_scaler.transform(input_data)

        # Make prediction using the Ridge model
        result = ridge_model.predict(scaled_data)

        # Return the prediction result
        return render_template("home.html", results=result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
