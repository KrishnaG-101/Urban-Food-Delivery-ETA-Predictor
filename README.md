# Urban Food Delivery ETA Predictor

End-to-end Machine Learning project to predict food delivery arrival times (ETA) based on delivery person characteristics, order details, traffic density, weather conditions, and spatial coordinates.

## Project Structure

```text
├── data/
│   └── raw/
│       ├── food_delivery.csv
│       └── test_dataset.csv
├── models/
│   └── .gitkeep
├── tests/
│   └── .gitkeep
├── Dockerfile
├── notebook.ipynb
├── predict.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── serve.py
└── train.py
```

## Quick Start & Environment Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, deterministic Python environment and dependency management.

### Prerequisites
- Python 3.11
- `uv` installed (`pip install uv` or via standalone installer)

### Setup with uv
```bash
# Create virtual environment and install dependencies
uv sync

# Or create virtual environment manually
uv venv --python 3.11
uv pip install -r requirements.txt
```

### Setup with standard pip
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the API Service
```bash
uv run uvicorn serve:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive Swagger documentation.

## Docker Support
```bash
docker build -t urban-food-delivery-eta .
docker run -p 8000:8000 urban-food-delivery-eta
```

