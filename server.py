# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import traceback

app = FastAPI()

# Allow the frontends you use for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoder
# Make sure these files exist and are valid (binary joblib/pickle)
model = joblib.load("mental_health_model.pkl")
country_encoder = joblib.load("country_encoder.pkl")

class RequestData(BaseModel):
    age: int
    gender: int
    country: str
    self_employed: int
    family_history: int
    work_interfere: int
    no_employees: int
    remote_work: int
    tech_company: int
    benefits: int
    care_options: int
    wellness_program: int
    seek_help: int
    anonymity: int
    leave: int
    mental_health_consequence: int
    phys_health_consequence: int
    coworkers: int
    supervisor: int
    mental_health_interview: int
    phys_health_interview: int
    mental_vs_physical: int
    obs_consequence: int

@app.post("/predict")
def predict(data: RequestData):
    try:
        # 1) Encode country. Be explicit and return a helpful error if unknown.
        try:
            enc = country_encoder.transform([data.country])
            # enc may be array-like (e.g., array([12])) or nested -> extract numeric
            country_encoded = enc[0]
            if isinstance(country_encoded, (list, tuple, np.ndarray)):
                country_encoded = country_encoded[0]
            if hasattr(country_encoded, "item"):
                country_encoded = country_encoded.item()
            country_encoded = float(country_encoded)
        except Exception as ex:
            # Return a descriptive error so frontend can show it
            return {"error": f"Country encoding failed for '{data.country}': {str(ex)}"}

        # 2) Arrange features in the exact order used for training:
        input_features = [
            data.age,
            data.gender,
            country_encoded,
            data.self_employed,
            data.family_history,
            data.work_interfere,
            data.no_employees,
            data.remote_work,
            data.tech_company,
            data.benefits,
            data.care_options,
            data.wellness_program,
            data.seek_help,
            data.anonymity,
            data.leave,
            data.mental_health_consequence,
            data.phys_health_consequence,
            data.coworkers,
            data.supervisor,
            data.mental_health_interview,
            data.phys_health_interview,
            data.mental_vs_physical,
            data.obs_consequence
        ]

        # 3) Convert to 2D array for the model
        arr = np.array(input_features, dtype=float).reshape(1, -1)

        # 4) Predict
        raw_pred = model.predict(arr)[0]

        # 5) Map predicted numeric to message
        try:
            pred_int = int(raw_pred)
        except Exception:
            pred_int = None

        # According to your setup: 0 -> mentally fit, 1 -> needs checkup
        if pred_int == 0:
            message = "✅ You are mentally fit."
        elif pred_int == 1:
            message = "⚠️ You might need a mental health check-up. Please consider speaking with a professional."
        else:
            message = f"Prediction returned unexpected value: {raw_pred}"

        return {"prediction": message}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Backend error: {str(e)}"}
