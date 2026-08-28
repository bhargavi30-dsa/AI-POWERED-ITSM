from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from gemini_service import analyze_incident
from feature_values import feature_values
from triage_agent import triage_agent
from triage.assignment_agent import intelligent_assignment
from sqlalchemy.orm import Session
from database import get_db, init_db, User, Ticket
from auth import hash_password, verify_password, create_access_token, get_current_user
from fastapi.security import OAuth2PasswordRequestForm
app = FastAPI(
    title="Enterprise AI ITSM",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()

# =========================================================
# LOAD MODELS
# =========================================================

category_model = joblib.load(
    r"D:\College_Projects\Enterprise_ITSM_AI\models_saved\category_prediction_pipeline.joblib"
)

category_encoder = joblib.load(
    r"D:\College_Projects\Enterprise_ITSM_AI\models_saved\category_label_encoder.joblib"
)

priority_model = joblib.load(
    r"D:\College_Projects\Enterprise_ITSM_AI\models_saved\priority_prediction_pipeline.joblib"
)

priority_encoder = joblib.load(
    r"D:\College_Projects\Enterprise_ITSM_AI\models_saved\priority_label_encoder.joblib"
)

sla_model = joblib.load(
    r"D:\College_Projects\Enterprise_ITSM_AI\models_saved\sla_prediction_pipeline.joblib"
)


# =========================================================
# INPUT SCHEMA
# =========================================================
class IncidentFeatures(BaseModel):
    contact_type: str
    location: str
    u_symptom: str
    impact: str
    urgency: str
    knowledge: bool
    notify: str
    opened_hour: int
    opened_day_of_week: int
    opened_month: int
    is_weekend: bool
class IncidentDescription(BaseModel):
    incident_description: str
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "employee"    

# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Enterprise AI ITSM Backend Running"
    }


# =========================================================
# CATEGORY PREDICTION
# =========================================================

@app.post("/predict-category")
def predict_category(data: IncidentFeatures):

    df = pd.DataFrame([data.model_dump()])

    prediction = category_model.predict(df)[0]

    prediction = category_encoder.inverse_transform(
        [prediction]
    )[0]

    return {
        "Predicted Category": prediction
    }


# =========================================================
# PRIORITY PREDICTION
# =========================================================

@app.post("/predict-priority")
def predict_priority(data: IncidentFeatures):

    df = pd.DataFrame([data.model_dump()])

    prediction = priority_model.predict(df)[0]

    prediction = priority_encoder.inverse_transform(
        [prediction]
    )[0]

    return {
        "Predicted Priority": prediction
    }


# =========================================================
# SLA PREDICTION
# =========================================================

@app.post("/predict-sla")
def predict_sla(data: IncidentFeatures):

    df = pd.DataFrame([data.model_dump()])

    prediction = sla_model.predict(df)[0]

    return {
        "Predicted SLA": bool(prediction)
    }
# =========================================================

@app.post("/analyze-incident")
def analyze_incident_endpoint(data: IncidentDescription):

    result = analyze_incident(data.incident_description)

    return result
# =========================================================
class TriageRequest(BaseModel):
    incident_description: str



@app.post("/complete-triage")
def complete_triage(
    data: TriageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    # ---- AGENT 1: Triage (summary, features, category, priority, SLA) ----
    agent1_result = triage_agent.invoke({
        "incident_description": data.incident_description,
        "summary": "",
        "features": {},
        "category": "",
        "priority": "",
        "sla": False,
        "final_result": {}
    })

    triage = agent1_result["final_result"]

    # ---- AGENT 2: Assignment + Resolver, using Agent 1's validated features ----
    assignment_result = intelligent_assignment(
        triage["validated_features"],
        top_n=3
    )

    # ---- SAVE TO DATABASE ----
    new_ticket = Ticket(
        user_id=current_user.id,
        incident_description=triage["incident_description"],
        summary=triage["summary"],
        predicted_category=triage["predicted_category"],
        predicted_priority=triage["predicted_priority"],
        predicted_sla=triage["predicted_sla"],
        predicted_assignment_group=assignment_result["predicted_assignment_group"],
        recommended_resolver=assignment_result["recommended_resolver"],
    )
    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)

    # ---- COMBINE ----
    return {
        "ticket_id": new_ticket.id,
        "incident_description": triage["incident_description"],
        "summary": triage["summary"],
        "validated_features": triage["validated_features"],
        "predicted_category": triage["predicted_category"],
        "predicted_priority": triage["predicted_priority"],
        "predicted_sla": triage["predicted_sla"],
        "predicted_assignment_group": assignment_result["predicted_assignment_group"],
        "recommended_resolver": assignment_result["recommended_resolver"],
        "alternative_resolvers": assignment_result["alternative_resolvers"],
        "resolver_history_available": assignment_result["resolver_history_available"]
    }
# FEATURE VALUES
# =========================================================

@app.get("/feature-values")
def get_feature_values():

    return feature_values

@app.post("/auth/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):

    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        name=data.name,
        email=data.email,
        hashed_password=hash_password(data.password),
        role=data.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Account created successfully", "user_id": new_user.id}
@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(data={"sub": str(user.id)})

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name
    }