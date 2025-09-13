# Car Price Prediction App
**[st126235]Dechathon's Machine Learning Assignment 1: Predicting Car Price**

Predict used car prices with a simple web interface powered by XGBoost.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dniamsaard4codework/Predicting-Car-Prices.git

# Change directory to the project
cd Predicting-Car-Prices

# Start the application
docker-compose up --build -d

# Open in browser
open http://localhost:8050
```

## How to Use

1. Fill in car details (year, mileage, engine size, etc.)
2. Click "Predict Price"
3. Get instant price estimate

## Stop the App

```bash
docker-compose down
```

## What's Inside

- **Web App**: Interactive dashboard with input forms (`app/app.py`)
- **ML Model**: Pre-trained XGBoost regression model (`model/car_price.model`)
- **Dataset**: Used car sales data for training (`data/Cars.csv`)
- **Notebook**: Complete model development & analysis (`notebook/st126235_Assignment_1.ipynb`)

## Tech Used

**ML**: Python 3.11 • XGBoost • Scikit-learn  
**Web**: Dash • Plotly  
**Deploy**: Docker • Docker Compose
