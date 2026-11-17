"""Urban Food Delivery ETA Predictor - Inference & Shared Feature Engineering Module.

Provides deterministic cleaning, feature engineering, and inference utilities
ensuring zero training-serving skew between notebook, training, and production API.
"""

from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Paths & Constants
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.bin"
MODEL_VERSION = "1.0.0"
DEFAULT_RMSE_WINDOW = 3.9  # Based on 5-fold CV & validation RMSE


def haversine_distance(
    lat1: Union[float, pd.Series],
    lon1: Union[float, pd.Series],
    lat2: Union[float, pd.Series],
    lon2: Union[float, pd.Series],
) -> Union[float, pd.Series]:
  """Calculates great-circle Haversine distance in kilometers between two GPS coordinates."""
  lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
      np.radians, [lon1, lat1, lon2, lat2]
  )
  dlon = lon2_rad - lon1_rad
  dlat = lat2_rad - lat1_rad
  a = (
      np.sin(dlat / 2.0) ** 2
      + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
  )
  c = 2.0 * np.arcsin(np.sqrt(a))
  return 6371.0 * c


def manhattan_distance(
    lat1: Union[float, pd.Series],
    lon1: Union[float, pd.Series],
    lat2: Union[float, pd.Series],
    lon2: Union[float, pd.Series],
) -> Union[float, pd.Series]:
  """Calculates approximate city-block (Manhattan L1) distance in kilometers."""
  lat_dist = np.abs(lat2 - lat1) * 111.0
  mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
  lon_dist = np.abs(lon2 - lon1) * 111.0 * np.cos(mean_lat_rad)
  return lat_dist + lon_dist


def clean_and_engineer(
    df_raw: pd.DataFrame, is_training: bool = False
) -> pd.DataFrame:
  """Performs deterministic cleaning and domain feature engineering on food delivery data.

  Ensures exact parity across training and inference pipelines.

  Parameters:
  -----------
  df_raw : pd.DataFrame
      Raw input records (single record or full dataset).
  is_training : bool, default False
      If True, enforces strict geospatial bounding box [8, 38] lat and [68, 98]
      lon
      to eliminate invalid coordinates (e.g. (0,0) Null Island) and parses
      target.

  Returns:
  --------
  pd.DataFrame
      Engineered feature matrix matching pipeline expectations.
  """
  df = df_raw.copy()

  # 1. Strip whitespace for string/object columns
  for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
      df[col] = df[col].astype(str).str.strip()

  # 2. Standardize missing sentinels to np.nan
  df = df.replace(
      {"NaN": np.nan, "conditions NaN": np.nan, "": np.nan, "None": np.nan}
  )

  # 3. Strip 'conditions ' prefix from Weatherconditions if present
  if "Weatherconditions" in df.columns:
    df["Weatherconditions"] = df["Weatherconditions"].apply(
        lambda x: re.sub(r"^conditions\s+", "", str(x)) if pd.notna(x) else x
    )

  # 4. Target variable extraction for training
  if "Time_taken(min)" in df.columns:
    df["time_taken_min"] = (
        df["Time_taken(min)"]
        .astype(str)
        .str.replace("(min)", "", regex=False)
        .str.strip()
    )
    df["time_taken_min"] = pd.to_numeric(df["time_taken_min"], errors="coerce")
    df = df.drop(columns=["Time_taken(min)"])

  # 5. Type casting for numeric fields
  numeric_cols = [
      "Delivery_person_Age",
      "Delivery_person_Ratings",
      "multiple_deliveries",
      "Vehicle_condition",
  ]
  for col in numeric_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # 6. Coordinate validation and correction (fix negative sign errors via abs)
  coord_cols = [
      "Restaurant_latitude",
      "Restaurant_longitude",
      "Delivery_location_latitude",
      "Delivery_location_longitude",
  ]
  for c in coord_cols:
    if c in df.columns:
      df[c] = pd.to_numeric(df[c], errors="coerce").abs()

  # In training mode, drop invalid / Null Island coordinates
  if is_training:
    valid_coords = (
        df["Restaurant_latitude"].between(8.0, 38.0)
        & df["Restaurant_longitude"].between(68.0, 98.0)
        & df["Delivery_location_latitude"].between(8.0, 38.0)
        & df["Delivery_location_longitude"].between(68.0, 98.0)
    )
    df = df[valid_coords].reset_index(drop=True)

  # 7. Compute physical distances
  if all(c in df.columns for c in coord_cols):
    df["distance_km"] = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    df["manhattan_distance_km"] = manhattan_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
  else:
    df["distance_km"] = np.nan
    df["manhattan_distance_km"] = np.nan

  # 8. City hub code extraction from Delivery_person_ID
  if "Delivery_person_ID" in df.columns:

    def extract_city(val: Any) -> str:
      if pd.isna(val) or not str(val).strip():
        return "UNKNOWN"
      val_str = str(val).strip()
      if "RES" in val_str:
        return re.split("RES", val_str)[0]
      return val_str[:4] if len(val_str) >= 4 else val_str

    df["city_code"] = df["Delivery_person_ID"].apply(extract_city)
  else:
    df["city_code"] = "UNKNOWN"

  # 9. Temporal features (order_hour & is_peak_hour)
  if "Time_Orderd" in df.columns:
    t_order = pd.to_datetime(
        df["Time_Orderd"], format="%H:%M:%S", errors="coerce"
    )
    df["order_hour"] = t_order.dt.hour
    df["is_peak_hour"] = df["order_hour"].apply(
        lambda h: (
            1
            if ((12 <= h <= 14) or (19 <= h <= 22))
            else (np.nan if pd.isna(h) else 0)
        )
    )
  else:
    df["order_hour"] = np.nan
    df["is_peak_hour"] = np.nan

  # 10. Kitchen preparation latency (prep_time_min) with midnight rollover
  if ("Time_Orderd" in df.columns) and ("Time_Order_picked" in df.columns):
    td_order = pd.to_timedelta(df["Time_Orderd"], errors="coerce")
    td_pick = pd.to_timedelta(df["Time_Order_picked"], errors="coerce")
    diff_min = (td_pick - td_order).dt.total_seconds() / 60.0
    # Add 1440 min (24h) if picked < ordered due to midnight crossing
    diff_min = np.where(diff_min < 0, diff_min + 1440, diff_min)
    diff_min = np.where(td_order.isna() | td_pick.isna(), np.nan, diff_min)
    # Cap / null invalid extreme or negative values
    diff_min = np.where((diff_min < 0) | (diff_min > 60), np.nan, diff_min)
    df["prep_time_min"] = diff_min
  else:
    df["prep_time_min"] = np.nan

  # 11. Calendar features (day_of_week & is_weekend)
  if "Order_Date" in df.columns:
    dates = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    df["day_of_week"] = dates.dt.day_name().fillna("UNKNOWN")
    df["is_weekend"] = dates.dt.day_of_week.apply(
        lambda d: 1.0 if d >= 5 else (0.0 if pd.notna(d) else np.nan)
    )
  else:
    df["day_of_week"] = "UNKNOWN"
    df["is_weekend"] = np.nan

  # 12. Traffic density score and courier interaction
  traffic_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Jam": 4.0}
  if "Road_traffic_density" in df.columns:
    df["traffic_density_score"] = df["Road_traffic_density"].map(traffic_map)
  else:
    df["traffic_density_score"] = np.nan

  if (
      "Delivery_person_Ratings" in df.columns
      and "traffic_density_score" in df.columns
  ):
    df["ratings_x_traffic"] = (
        df["Delivery_person_Ratings"] * df["traffic_density_score"]
    )
  else:
    df["ratings_x_traffic"] = np.nan

  # 13. Drop raw identifier and unparsed timestamp strings
  cols_to_drop = [
      "ID",
      "Delivery_person_ID",
      "Order_Date",
      "Time_Orderd",
      "Time_Order_picked",
  ]
  df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

  return df


@lru_cache(maxsize=1)
def load_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
  """Loads and caches the serialized model bundle."""
  path = Path(model_path)
  if not path.exists():
    raise FileNotFoundError(
        f"Model artifact not found at '{path}'. Please run 'python train.py'"
        " first."
    )
  bundle = joblib.load(path)
  return bundle


def predict_eta(
    order_data: Dict[str, Any], model_bundle: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
  """Transforms a single order dictionary and returns the predicted ETA with confidence window.

  Parameters:
  -----------
  order_data : dict
      Dictionary representing a single food delivery order.
  model_bundle : dict, optional
      Pre-loaded model bundle; loads DEFAULT_MODEL_PATH if None.

  Returns:
  --------
  dict:
      {
          "predicted_eta_min": float,
          "estimated_window_min": [float, float],
          "model_version": str
      }
  """
  if model_bundle is None:
    model_bundle = load_model()

  pipeline = model_bundle["pipeline"]
  feature_cols = model_bundle.get("feature_columns")
  version = model_bundle.get("version", MODEL_VERSION)

  # Process single-record input through shared feature engineering
  raw_df = pd.DataFrame([order_data])
  engineered_df = clean_and_engineer(raw_df, is_training=False)

  # Ensure all expected feature columns are present
  if feature_cols:
    for col in feature_cols:
      if col not in engineered_df.columns:
        engineered_df[col] = np.nan
    engineered_df = engineered_df[feature_cols]

  # Generate point prediction
  pred = float(pipeline.predict(engineered_df)[0])
  predicted_eta = round(pred, 2)
  lower_bound = max(5.0, round(predicted_eta - DEFAULT_RMSE_WINDOW, 1))
  upper_bound = round(predicted_eta + DEFAULT_RMSE_WINDOW, 1)

  return {
      "predicted_eta_min": predicted_eta,
      "estimated_window_min": [lower_bound, upper_bound],
      "model_version": version,
  }


if __name__ == "__main__":
  print("=" * 70)
  print("Urban Food Delivery ETA Predictor - Standalone CLI Smoke Test")
  print("=" * 70)

  sample_order = {
      "Delivery_person_ID": "BANGRES19DEL01",
      "Delivery_person_Age": 29.0,
      "Delivery_person_Ratings": 4.8,
      "Restaurant_latitude": 12.914264,
      "Restaurant_longitude": 77.678400,
      "Delivery_location_latitude": 12.924264,
      "Delivery_location_longitude": 77.688400,
      "Order_Date": "19-03-2022",
      "Time_Orderd": "19:30:00",
      "Time_Order_picked": "19:40:00",
      "Weatherconditions": "Sunny",
      "Road_traffic_density": "Jam",
      "Vehicle_condition": 2,
      "Type_of_order": "Meal",
      "Type_of_vehicle": "motorcycle",
      "multiple_deliveries": 1.0,
      "Festival": "No",
      "City": "Metropolitian",
  }

  print("\nSample Order Input:")
  for k, v in sample_order.items():
    print(f"  {k:<28}: {v}")

  try:
    result = predict_eta(sample_order)
    print("\nInference Output:")
    print(f"  Predicted ETA:       {result['predicted_eta_min']} minutes")
    print(
        "  Estimated Window:   "
        f" {result['estimated_window_min'][0]} - {result['estimated_window_min'][1]} minutes"
    )
    print(f"  Model Version:       {result['model_version']}")
    print("=" * 70)
  except FileNotFoundError as err:
    print(f"\n[!] {err}")
