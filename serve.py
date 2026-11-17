"""Urban Food Delivery ETA Predictor - Production FastAPI Web Service.

Exposes RESTful endpoints for real-time delivery time inference (ETA),
listening by default on 0.0.0.0:9696.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from predict import MODEL_VERSION, load_model, predict_eta

# Global model bundle holder
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Loads the model bundle on application startup."""
  try:
    app_state["model_bundle"] = load_model()
    print(f"[*] Production model bundle v{MODEL_VERSION} loaded successfully.")
  except Exception as e:
    print(f"[!] Warning: Model could not be loaded at startup: {e}")
    app_state["model_bundle"] = None
  yield
  app_state.clear()


app = FastAPI(
    title="Urban Food Delivery ETA Predictor",
    description=(
        "Production machine learning API predicting on-demand food delivery"
        " arrival times (ETA) based on rider characteristics, traffic density,"
        " weather, and spatial routing."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ==============================================================================
# Pydantic Schemas (Pydantic v2)
# ==============================================================================
class DeliveryOrderRequest(BaseModel):
  """Order payload schema representing a single food delivery dispatch request."""

  model_config = ConfigDict(
      populate_by_name=True,
      json_schema_extra={
          "example": {
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
      },
  )

  # Required Geospatial coordinates
  Restaurant_latitude: float = Field(
      ...,
      ge=-90.0,
      le=90.0,
      description="Latitude coordinate of restaurant pickup point",
  )
  Restaurant_longitude: float = Field(
      ...,
      ge=-180.0,
      le=180.0,
      description="Longitude coordinate of restaurant pickup point",
  )
  Delivery_location_latitude: float = Field(
      ...,
      ge=-90.0,
      le=90.0,
      description="Latitude coordinate of customer delivery destination",
  )
  Delivery_location_longitude: float = Field(
      ...,
      ge=-180.0,
      le=180.0,
      description="Longitude coordinate of customer delivery destination",
  )

  # Optional features (imputed by production pipeline if omitted)
  Delivery_person_ID: Optional[str] = Field(
      None, description="Courier ID prefix containing regional city code"
  )
  Delivery_person_Age: Optional[float] = Field(
      None, ge=16.0, le=80.0, description="Age of the delivery rider"
  )
  Delivery_person_Ratings: Optional[float] = Field(
      None, ge=1.0, le=5.0, description="Average rider customer rating (1.0-5.0)"
  )
  Order_Date: Optional[str] = Field(
      None, description="Order placement date formatted as DD-MM-YYYY"
  )
  Time_Orderd: Optional[str] = Field(
      None, description="Time order was confirmed by kitchen (HH:MM:SS)"
  )
  Time_Order_picked: Optional[str] = Field(
      None, description="Time order was picked up by rider (HH:MM:SS)"
  )
  Weatherconditions: Optional[str] = Field(
      None,
      description=(
          "Weather condition (Sunny, Stormy, Sandstorms, Windy, Cloudy, Fog)"
      ),
  )
  Road_traffic_density: Optional[str] = Field(
      None, description="Traffic congestion density (Low, Medium, High, Jam)"
  )
  Vehicle_condition: Optional[int] = Field(
      None, ge=0, le=3, description="Rider vehicle condition score (0 to 3)"
  )
  Type_of_order: Optional[str] = Field(
      None, description="Food order category (Snack, Meal, Drinks, Buffet)"
  )
  Type_of_vehicle: Optional[str] = Field(
      None,
      description=(
          "Delivery vehicle type (motorcycle, scooter, electric_scooter,"
          " bicycle)"
      ),
  )
  multiple_deliveries: Optional[float] = Field(
      None, ge=0.0, le=5.0, description="Number of additional concurrent orders"
  )
  Festival: Optional[str] = Field(
      None, description="Flag indicating active holiday/festival ('Yes' / 'No')"
  )
  City: Optional[str] = Field(
      None,
      description="Urbanization density (Urban, Metropolitian, Semi-Urban)",
  )


class PredictionResponse(BaseModel):
  """Response payload containing point prediction and confidence window."""

  model_config = ConfigDict(
      json_schema_extra={
          "example": {
              "predicted_eta_min": 24.52,
              "estimated_window_min": [20.6, 28.4],
              "model_version": "1.0.0",
          }
      }
  )

  predicted_eta_min: float = Field(
      ...,
      description=(
          "Point prediction for total delivery duration in minutes from"
          " order placement to doorstep"
      ),
  )
  estimated_window_min: List[float] = Field(
      ...,
      description=(
          "Estimated delivery duration interval [min, max] based on"
          " validation RMSE uncertainty"
      ),
  )
  model_version: str = Field(
      ..., description="Semantic version of the serving model artifact"
  )


class HealthResponse(BaseModel):
  """Health check schema."""

  status: str = Field(..., description="Service health status ('healthy')")
  model_loaded: bool = Field(
      ..., description="True if ML pipeline artifact is loaded and ready"
  )
  model_version: str = Field(
      ..., description="Loaded model version string ('1.0.0')"
  )


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
)
def health_check() -> HealthResponse:
  """Health check endpoint indicating whether the service and model are ready."""
  bundle = app_state.get("model_bundle")
  if bundle is None:
    try:
      bundle = load_model()
      app_state["model_bundle"] = bundle
    except Exception:
      bundle = None

  is_loaded = bundle is not None
  return HealthResponse(
      status="healthy", model_loaded=is_loaded, model_version=MODEL_VERSION
  )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(order: DeliveryOrderRequest) -> PredictionResponse:
  """Predicts the delivery time in minutes for an incoming food order."""
  bundle = app_state.get("model_bundle")
  if bundle is None:
    try:
      bundle = load_model()
      app_state["model_bundle"] = bundle
    except Exception as e:
      raise HTTPException(
          status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
          detail=(
              f"Model artifact not loaded: {e}. Please ensure models/model.bin"
              " exists."
          ),
      )

  try:
    order_dict = order.model_dump()
    result = predict_eta(order_dict, model_bundle=bundle)
    return PredictionResponse(**result)
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Inference error processing order: {str(e)}",
    )


if __name__ == "__main__":
  uvicorn.run("serve:app", host="0.0.0.0", port=9696, reload=False)
