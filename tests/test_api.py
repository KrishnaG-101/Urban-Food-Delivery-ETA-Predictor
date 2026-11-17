"""Automated Test Suite for Urban Food Delivery ETA Predictor API."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import app

client = TestClient(app)


def test_health_check():
  """Verifies GET /health endpoint returns 200 OK and model is ready."""
  response = client.get("/health")
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "healthy"
  assert data["model_loaded"] is True
  assert data["model_version"] == "1.0.0"


def test_predict_valid_standard_order():
  """Verifies POST /predict returns 200 OK and an ETA within realistic bounds (10-60 min)."""
  payload = {
      "Delivery_person_ID": "BANGRES19DEL01",
      "Delivery_person_Age": 28.0,
      "Delivery_person_Ratings": 4.8,
      "Restaurant_latitude": 12.914264,
      "Restaurant_longitude": 77.678400,
      "Delivery_location_latitude": 12.924264,
      "Delivery_location_longitude": 77.688400,
      "Order_Date": "19-03-2022",
      "Time_Orderd": "19:30:00",
      "Time_Order_picked": "19:40:00",
      "Weatherconditions": "Sunny",
      "Road_traffic_density": "Low",
      "Vehicle_condition": 2,
      "Type_of_order": "Meal",
      "Type_of_vehicle": "motorcycle",
      "multiple_deliveries": 0.0,
      "Festival": "No",
      "City": "Metropolitian",
  }

  response = client.post("/predict", json=payload)
  assert response.status_code == 200
  data = response.json()

  assert "predicted_eta_min" in data
  assert "estimated_window_min" in data
  assert data["model_version"] == "1.0.0"

  eta = data["predicted_eta_min"]
  assert 10.0 <= eta <= 60.0
  assert len(data["estimated_window_min"]) == 2
  assert data["estimated_window_min"][0] < data["estimated_window_min"][1]


def test_predict_traffic_and_weather_impact():
  """Verifies severe conditions (Jam traffic + Stormy weather + multiple deliveries)

  predict a significantly higher ETA than optimal conditions (Low traffic +
  Sunny + 0 deliveries).
  """
  low_stress_payload = {
      "Delivery_person_ID": "BANGRES19DEL01",
      "Delivery_person_Age": 26.0,
      "Delivery_person_Ratings": 4.9,
      "Restaurant_latitude": 12.914264,
      "Restaurant_longitude": 77.678400,
      "Delivery_location_latitude": 12.924264,
      "Delivery_location_longitude": 77.688400,
      "Order_Date": "19-03-2022",
      "Time_Orderd": "10:30:00",
      "Time_Order_picked": "10:40:00",
      "Weatherconditions": "Sunny",
      "Road_traffic_density": "Low",
      "Vehicle_condition": 3,
      "Type_of_order": "Snack",
      "Type_of_vehicle": "scooter",
      "multiple_deliveries": 0.0,
      "Festival": "No",
      "City": "Urban",
  }

  high_stress_payload = {
      "Delivery_person_ID": "BANGRES19DEL01",
      "Delivery_person_Age": 38.0,
      "Delivery_person_Ratings": 3.9,
      "Restaurant_latitude": 12.914264,
      "Restaurant_longitude": 77.678400,
      "Delivery_location_latitude": 12.924264,
      "Delivery_location_longitude": 77.688400,
      "Order_Date": "19-03-2022",
      "Time_Orderd": "20:30:00",
      "Time_Order_picked": "20:45:00",
      "Weatherconditions": "Stormy",
      "Road_traffic_density": "Jam",
      "Vehicle_condition": 0,
      "Type_of_order": "Meal",
      "Type_of_vehicle": "motorcycle",
      "multiple_deliveries": 2.0,
      "Festival": "Yes",
      "City": "Metropolitian",
  }

  res_low = client.post("/predict", json=low_stress_payload)
  res_high = client.post("/predict", json=high_stress_payload)

  assert res_low.status_code == 200
  assert res_high.status_code == 200

  eta_low = res_low.json()["predicted_eta_min"]
  eta_high = res_high.json()["predicted_eta_min"]

  # High stress delivery should take noticeably longer than low stress
  assert eta_high > eta_low
  assert eta_high - eta_low >= 8.0


def test_predict_missing_required_fields():
  """Verifies that omitting required coordinates triggers a 422 Unprocessable Entity."""
  invalid_payload = {
      "Delivery_person_Age": 28.0,
      "Weatherconditions": "Sunny",
      # Missing required Restaurant_latitude, Restaurant_longitude, Delivery_location_latitude, Delivery_location_longitude
  }

  response = client.post("/predict", json=invalid_payload)
  assert response.status_code == 422
  errors = response.json().get("detail", [])
  assert len(errors) > 0


def test_predict_handles_partial_optional_fields():
  """Verifies that providing only required coordinates and minimal fields works cleanly."""
  minimal_payload = {
      "Restaurant_latitude": 22.745049,
      "Restaurant_longitude": 75.892471,
      "Delivery_location_latitude": 22.765049,
      "Delivery_location_longitude": 75.912471,
  }

  response = client.post("/predict", json=minimal_payload)
  assert response.status_code == 200
  data = response.json()
  assert "predicted_eta_min" in data
  assert 10.0 <= data["predicted_eta_min"] <= 60.0
