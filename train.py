import pandas as pd
import numpy as np
import os
import joblib
import glob

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# LOAD DATA
# =========================
files = glob.glob("data/*.csv")

df_list = [pd.read_csv(f) for f in files]

df = pd.concat(df_list, ignore_index=True)

print("Total data:", len(df))


# =========================
# FEATURE & TARGET
# =========================

X = df[['latitude','longitude']]

y_mag = df['magnitude']
y_depth = df['depth']


# =========================
# TRAIN TEST SPLIT (80:20)
# =========================

X_train, X_test, y_mag_train, y_mag_test = train_test_split(
    X, y_mag,
    test_size=0.20,
    random_state=42
)

_, _, y_depth_train, y_depth_test = train_test_split(
    X, y_depth,
    test_size=0.20,
    random_state=42
)

print("Training data:", len(X_train))
print("Testing data :", len(X_test))


# =========================
# MODEL PARAMETER
# =========================

rf_mag = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_depth = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# =========================
# TRAIN MODEL
# =========================

print("\nTraining magnitude model...")
rf_mag.fit(X_train, y_mag_train)

print("Training depth model...")
rf_depth.fit(X_train, y_depth_train)


# =========================
# PREDICTION
# =========================

pred_mag = rf_mag.predict(X_test)
pred_depth = rf_depth.predict(X_test)


# =========================
# EVALUATION METRICS
# =========================

def evaluate(y_true, y_pred, name):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n===== Evaluation:", name, "=====")
    print("MAE :", round(mae,4))
    print("RMSE:", round(rmse,4))
    print("R2  :", round(r2,4))

    return mae, rmse, r2


evaluate(y_mag_test, pred_mag, "Magnitude")
evaluate(y_depth_test, pred_depth, "Depth")


# =========================
# CROSS VALIDATION
# =========================

print("\nRunning 5-Fold Cross Validation...")

cv_mag = cross_val_score(
    rf_mag,
    X,
    y_mag,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

cv_depth = cross_val_score(
    rf_depth,
    X,
    y_depth,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCV R2 Magnitude :", round(cv_mag.mean(),4))
print("CV R2 Depth     :", round(cv_depth.mean(),4))


# =========================
# SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(rf_mag, "models/rf_magnitude.pkl")
joblib.dump(rf_depth, "models/rf_depth.pkl")

print("\nModel saved in folder /models")