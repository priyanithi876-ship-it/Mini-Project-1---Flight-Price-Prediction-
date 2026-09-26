# ============================================================
# FLIGHT PRICE PREDICTION
# TRAIN MODEL AND CREATE model.joblib
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("=" * 60)
print("FLIGHT PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("flightprice.csv", sep=None, engine="python")

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. REMOVE UNNECESSARY COLUMNS
# ------------------------------------------------------------

if "unnamed:_0" in df.columns:
    df = df.drop(columns=["unnamed:_0"])

if "flight" in df.columns:
    df = df.drop(columns=["flight"])


# ------------------------------------------------------------
# 4. CLEAN TARGET
# ------------------------------------------------------------

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df = df.dropna(subset=["price"])


# ------------------------------------------------------------
# 5. CONVERT DURATION TO HOURS
# ------------------------------------------------------------

if "duration" in df.columns:

    def convert_duration(value):

        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower()

        try:
            if "h" in value:

                hours = 0
                minutes = 0

                parts = value.split()

                for part in parts:

                    if part.endswith("h"):
                        hours = float(part[:-1])

                    elif part.endswith("m"):
                        minutes = float(part[:-1])

                return hours + minutes / 60

            elif "m" in value:

                return float(value.replace("m", "").strip()) / 60

            else:

                return float(value)

        except:
            return np.nan

    df["duration"] = df["duration"].apply(convert_duration)


# ------------------------------------------------------------
# 6. REMOVE DUPLICATES
# ------------------------------------------------------------

df = df.drop_duplicates()

print("\nCleaned dataset shape:", df.shape)


# ------------------------------------------------------------
# 7. SEPARATE X AND y
# ------------------------------------------------------------

X = df.drop(columns=["price"])
y = df["price"]


# ------------------------------------------------------------
# 8. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ------------------------------------------------------------

numeric_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ------------------------------------------------------------
# 9. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ------------------------------------------------------------
# 10. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ------------------------------------------------------------
# 11. CREATE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ------------------------------------------------------------
# 12. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ------------------------------------------------------------
# 13. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 14. EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 4))


# ------------------------------------------------------------
# 15. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(
    model,
    "model.joblib"
)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nCreated file:")
print("model.joblib")

print("\nYou can now use model.joblib in app.py.")
