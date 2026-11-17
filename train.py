"""Urban Food Delivery ETA Predictor - Model Training & Artifact Serialization Pipeline.

Reproduces the winning LightGBM pipeline from notebook.ipynb identically,
trains on full train (train + val, 80% data), evaluates on held-out test (20% data),
and exports the serialized bundle to models/model.bin.
"""

from pathlib import Path
import sys
import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import shared deterministic feature engineering from predict.py
from predict import clean_and_engineer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "food_delivery.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.bin"
MODEL_VERSION = "1.0.0"

# Exact winning hyperparameters discovered via 5-fold CV RandomizedSearchCV
WINNING_LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 50,
    "max_depth": 12,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def train_model() -> None:
  """Executes the complete training pipeline and exports models/model.bin."""
  print("=" * 80)
  print("URBAN FOOD DELIVERY ETA PREDICTOR - PRODUCTION TRAINING PIPELINE")
  print("=" * 80)

  if not DATA_PATH.exists():
    raise FileNotFoundError(f"Raw data file not found at: {DATA_PATH}")

  print(f"[1/6] Loading raw dataset from: {DATA_PATH}")
  raw_df = pd.read_csv(DATA_PATH)
  print(f"      Loaded raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

  print("[2/6] Applying deterministic data cleaning and domain feature engineering...")
  features_df = clean_and_engineer(raw_df, is_training=True)
  print(
      f"      Engineered shape: {features_df.shape[0]:,} rows x"
      f" {features_df.shape[1]} columns"
  )

  print("[3/6] Reproducing 60/20/20 data split (random_state=42)...")
  train_val_df, df_test = train_test_split(
      features_df, test_size=0.20, random_state=42
  )
  df_train, df_val = train_test_split(
      train_val_df, test_size=0.25, random_state=42
  )

  # Full training set combines train + val (80% of data)
  df_full_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
  X_full_train = df_full_train.drop(columns=["time_taken_min"])
  y_full_train = df_full_train["time_taken_min"]

  X_test = df_test.drop(columns=["time_taken_min"])
  y_test = df_test["time_taken_min"]

  print(
      f"      df_full_train (Train + Val): {X_full_train.shape[0]:,} rows"
      f" ({len(X_full_train) / len(features_df):.1%})"
  )
  print(
      f"      df_test (Held-Out Test):     {X_test.shape[0]:,} rows"
      f" ({len(X_test) / len(features_df):.1%})"
  )

  feature_cols = X_full_train.columns.tolist()
  num_features = X_full_train.select_dtypes(include=[np.number]).columns.tolist()
  cat_features = X_full_train.select_dtypes(
      include=["object", "category", "string"]
  ).columns.tolist()

  print(
      f"[4/6] Assembling scikit-learn pipeline with {len(num_features)} numeric"
      f" and {len(cat_features)} categorical features..."
  )
  numeric_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
      ]
  )

  categorical_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
      ]
  )

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, num_features),
          ("cat", categorical_transformer, cat_features),
      ],
      verbose_feature_names_out=False,
  )

  production_pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("regressor", LGBMRegressor(**WINNING_LGBM_PARAMS)),
      ]
  )

  print(
      "[5/6] Fitting full production pipeline on df_full_train (33,562 rows)..."
  )
  production_pipeline.fit(X_full_train, y_full_train)
  print("      Pipeline training complete.")

  print("[6/6] Evaluating on held-out df_test (8,391 rows)...")
  y_pred = production_pipeline.predict(X_test)
  test_rmse = float(root_mean_squared_error(y_test, y_pred))
  test_mae = float(mean_absolute_error(y_test, y_pred))
  test_r2 = float(r2_score(y_test, y_pred))

  print("=" * 80)
  print("FINAL TEST EVALUATION METRICS:")
  print(f"  Test RMSE:  {test_rmse:.4f} minutes")
  print(f"  Test MAE:   {test_mae:.4f} minutes")
  print(f"  Test R2:    {test_r2:.4f}")
  print("=" * 80)

  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  artifact_bundle = {
      "pipeline": production_pipeline,
      "feature_columns": feature_cols,
      "metrics": {
          "test_rmse": round(test_rmse, 4),
          "test_mae": round(test_mae, 4),
          "test_r2": round(test_r2, 4),
      },
      "version": MODEL_VERSION,
  }

  joblib.dump(artifact_bundle, MODEL_PATH)
  print(
      f"Model artifact bundle successfully exported to: {MODEL_PATH} ("
      f"{MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB)"
  )
  print("=" * 80)


if __name__ == "__main__":
  train_model()
