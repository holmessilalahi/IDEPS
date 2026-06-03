from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# GRAPH STYLE
plt.style.use('seaborn-v0_8-whitegrid')

# LOAD TRAINED MODELS
model_mag = joblib.load("models/rf_magnitude.pkl")
model_depth = joblib.load("models/rf_depth.pkl")

print("Models loaded successfully")


# CREATE PLOT FUNCTION
def create_plot(data, predicted_value, label):

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        data,
        bins=20,
        stat="probability",
        color="#174a8b",
        edgecolor="white",
        kde=True,
        ax=ax
    )

    ax.axvline(
        predicted_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Predicted = {predicted_value:.2f}"
    )

    ax.set_title(
        f"Probability Distribution of {label}",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(label)
    ax.set_ylabel("Probability")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.legend()

    plt.tight_layout()

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    buf.seek(0)

    img_base64 = base64.b64encode(
        buf.read()
    ).decode("utf-8")

    plt.close(fig)

    return img_base64


# HOME PAGE
@app.route('/')
def index():
    return render_template('index.html')


# PREDICTION API
@app.route('/predict', methods=['POST'])
def predict():

    try:

        lat = float(request.form['latitude'])
        lon = float(request.form['longitude'])

        # Basic validation
        if not (-90 <= lat <= 90):
            return jsonify({
                "error": "Latitude must be between -90 and 90"
            })

        if not (-180 <= lon <= 180):
            return jsonify({
                "error": "Longitude must be between -180 and 180"
            })

        X_input = np.array([[lat, lon]])

        # Mean prediction
        mag_pred = model_mag.predict(X_input)[0]
        depth_pred = model_depth.predict(X_input)[0]

        # Distribution from all trees
        mag_all = [
            tree.predict(X_input)[0]
            for tree in model_mag.estimators_
        ]

        depth_all = [
            tree.predict(X_input)[0]
            for tree in model_depth.estimators_
        ]

        # Confidence interval (95%)
        mag_p05 = np.percentile(mag_all, 2.5)
        mag_p95 = np.percentile(mag_all, 97.5)

        depth_p05 = np.percentile(depth_all, 2.5)
        depth_p95 = np.percentile(depth_all, 97.5)

        # Create plots
        mag_plot = create_plot(
            mag_all,
            mag_pred,
            "Magnitude"
        )

        depth_plot = create_plot(
            depth_all,
            depth_pred,
            "Depth (km)"
        )

        return jsonify({

            "magnitude": round(float(mag_pred), 2),

            "depth": round(float(depth_pred), 2),

            "magnitude_ci": [
                round(float(mag_p05), 2),
                round(float(mag_p95), 2)
            ],

            "depth_ci": [
                round(float(depth_p05), 2),
                round(float(depth_p95), 2)
            ],

            "mag_plot": mag_plot,

            "depth_plot": depth_plot

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# RUN APP
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )