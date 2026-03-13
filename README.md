# Multi-Modal AI System: Vehicle Intelligence Platform

## Overview

This project builds a **Multi-Modal AI System** that analyzes multiple
inputs to automatically generate a **vehicle service record**.

The system processes: - Vehicle images - Customer text requests -
Vehicle metadata

and produces structured insights about the vehicle and service priority.

------------------------------------------------------------------------

## Features

### Vehicle Type Detection

Detects the vehicle type from an image (car, bike, truck, SUV).

### Damage Detection

Identifies visible damages such as: - Dent - Scratch - Broken glass

### Customer Intent Detection

Extracts customer intent from text requests.

Example: "My car has a dent and I need insurance support."

Detected intent: "insurance claim"

### Multi-Modal Fusion

Combines: - Image analysis - Customer request - Vehicle metadata

to generate a structured service record.

------------------------------------------------------------------------

## Example Output

``` json
{
  "vehicle_type": "SUV",
  "detected_damage": ["dent"],
  "customer_intent": "insurance claim",
  "service_priority": "high"
}
```

------------------------------------------------------------------------

## Run the API

Install dependencies:

    pip install -r requirements.txt

Start the FastAPI server:

    python -m uvicorn app.main:app --reload --port 8000

Open API documentation:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Project Structure

    Multi-Model-AI-System
    │
    ├── app
    │   ├── main.py
    │   ├── inference.py
    │   ├── fusion.py
    │   └── schemas.py
    │
    ├── scripts
    ├── config
    ├── sample_data
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Datasets

Datasets are not included due to size. They can be downloaded from
Kaggle datasets mentioned in the assignment instructions.

------------------------------------------------------------------------

## Technologies Used

-   Python
-   FastAPI
-   PyTorch
-   Scikit-Learn
-   Pandas
-   NumPy

------------------------------------------------------------------------

## Author

Anish Kumar
