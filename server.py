from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("mental_health_model.pkl", "rb") as f:
    model = joblib.load(f)
# Load encoder
with open("country_encoder.pkl", "rb") as f:
    country_encoder = joblib.load(f)

# Request schema
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
        # Encode country
        country_encoded = country_encoder.transform([data.country])[0]

        # Prepare input features
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

        # Get prediction (0 or 1)
        prediction = model.predict([input_features])[0]

        # Map prediction to user message
        if prediction == 0:
            message = "You might need a mental health check-up."
        elif prediction == 1:
            message = "You seem mentally well, no treatment needed."
        else:
            message = "Unable to determine mental health status."

        # Return formatted response
        return {"prediction": message}

    except Exception as e:
        return {"error": f"Backend error: {str(e)}"}
