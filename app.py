from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import glob

app = Flask(__name__)

# GRAPH STYLE
plt.style.use('seaborn-v0_8-whitegrid')

# LOAD DATA
files = glob.glob("data/*.csv")

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

print("Total data:", len(df))

X = df[['latitude','longitude']]
y_mag = df['magnitude']
y_depth = df['depth']


# TRAIN MODEL
model_mag = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_depth = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_mag.fit(X, y_mag)
model_depth.fit(X, y_depth)


# FUNCTION CREATE GRAPH
def create_plot(data, predicted_value, label):

    fig, ax = plt.subplots(figsize=(6,4))

    # Histogram probability
    sns.histplot(
        data,
        bins=20,
        stat="probability",
        color="#174a8b",
        edgecolor="white",
        kde=True,
        ax=ax
    )

    # Predicted value line
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
        axis='y',
        linestyle="--",
        alpha=0.4
    )

    ax.legend()

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    plt.close()

    return img_base64


# ROUTE
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    lat = float(request.form['latitude'])
    lon = float(request.form['longitude'])

    X_input = np.array([[lat, lon]])

    # Mean prediction
    mag_pred = model_mag.predict(X_input)[0]
    depth_pred = model_depth.predict(X_input)[0]

    # Prediction distribution from RF trees
    mag_all = [tree.predict(X_input)[0] for tree in model_mag.estimators_]
    depth_all = [tree.predict(X_input)[0] for tree in model_depth.estimators_]

    # Create plots
    mag_plot = create_plot(mag_all, mag_pred, "Magnitude")
    depth_plot = create_plot(depth_all, depth_pred, "Depth (km)")

    return jsonify({
        "magnitude": round(mag_pred,2),
        "depth": round(depth_pred,2),
        "mag_plot": mag_plot,
        "depth_plot": depth_plot
    })


# RUN APP
if __name__ == '__main__':
    app.run(debug=True)