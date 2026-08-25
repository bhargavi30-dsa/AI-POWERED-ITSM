import os
import json
import time
from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from feature_values import feature_values


# ============================================================
# 1. PROJECT PATH
# ============================================================

# File location:
# D:\Enterprise_ITSM_AI\backend\triage_agent.py
#
# Therefore:
# parent       = backend
# parent.parent = Enterprise_ITSM_AI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models_saved"


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================================
# 3. GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# ============================================================
# 4. SAFE GEMINI INVOCATION
# ============================================================

def safe_llm_invoke(prompt, retries=2):
    """
    Safely calls Gemini.

    If Gemini temporarily returns a 429 / RESOURCE_EXHAUSTED
    error, the function waits and retries.

    IMPORTANT:
    If the free-tier daily quota is exhausted, waiting
    will not necessarily solve the problem.
    """

    for attempt in range(retries + 1):

        try:
            return llm.invoke(prompt)

        except Exception as e:

            error_text = str(e)

            # --------------------------------------------
            # Gemini rate-limit / quota error
            # --------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                # Default wait
                wait_time = 10

                # ----------------------------------------
                # Try to extract Google's retry time
                # ----------------------------------------

                try:

                    if "Please retry in" in error_text:

                        text = error_text.split(
                            "Please retry in"
                        )[1]

                        seconds_text = (
                            text.split("s")[0].strip()
                        )

                        seconds = float(seconds_text)

                        wait_time = max(
                            5,
                            int(seconds) + 1
                        )

                except Exception:
                    wait_time = 10

                # ----------------------------------------
                # Retry
                # ----------------------------------------

                if attempt < retries:

                    print(
                        "\nGemini rate limit reached."
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    print(
                        f"Attempt {attempt + 1}/{retries}"
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        "\n===================================="
                    )
                    print(
                        "GEMINI QUOTA / RATE LIMIT EXHAUSTED"
                    )
                    print(
                        "===================================="
                    )

                    print(
                        "Gemini rejected the request because "
                        "the API quota/rate limit was exceeded."
                    )

                    print(
                        "If this is a daily free-tier quota, "
                        "waiting a few seconds will not reset it."
                    )

                    print(
                        "Check your Gemini API usage/billing "
                        "or use another available API key/model."
                    )

                    print(
                        "===================================="
                    )

                    raise

            else:

                # Any non-quota error should immediately
                # be raised.
                raise


# ============================================================
# 5. CHECK MODEL FILES
# ============================================================

print("\n====================================")
print("CHECKING ML MODEL FILES")
print("====================================")

print(f"Project root : {PROJECT_ROOT}")
print(f"Models dir   : {MODELS_DIR}")


required_model_files = [

    MODELS_DIR
    / "category_prediction_pipeline.joblib",

    MODELS_DIR
    / "category_label_encoder.joblib",

    MODELS_DIR
    / "priority_prediction_pipeline.joblib",

    MODELS_DIR
    / "priority_label_encoder.joblib",

    MODELS_DIR
    / "sla_prediction_pipeline.joblib"
]


for model_file in required_model_files:

    if not model_file.exists():

        raise FileNotFoundError(
            f"\nML model file not found:\n{model_file}\n\n"
            f"Please check your models_saved folder."
        )

    print(f"Found: {model_file.name}")


print(
    "====================================\n"
)


# ============================================================
# 6. LOAD ML MODELS
# ============================================================


# ------------------------------------------------------------
# CATEGORY MODEL
# ------------------------------------------------------------

category_model = joblib.load(
    MODELS_DIR
    / "category_prediction_pipeline.joblib"
)

category_encoder = joblib.load(
    MODELS_DIR
    / "category_label_encoder.joblib"
)


# ------------------------------------------------------------
# PRIORITY MODEL
# ------------------------------------------------------------

priority_model = joblib.load(
    MODELS_DIR
    / "priority_prediction_pipeline.joblib"
)

priority_encoder = joblib.load(
    MODELS_DIR
    / "priority_label_encoder.joblib"
)


# ------------------------------------------------------------
# SLA MODEL
# ------------------------------------------------------------

sla_model = joblib.load(
    MODELS_DIR
    / "sla_prediction_pipeline.joblib"
)


# ============================================================
# 7. LANGGRAPH STATE
# ============================================================

class TriageState(TypedDict):

    incident_description: str

    summary: str

    features: dict

    category: str

    priority: str

    sla: bool

    final_result: dict


# ============================================================
# 8. NODE 1
#    ANALYZE INCIDENT + EXTRACT FEATURES
# ============================================================

def analyze_and_extract(
    state: TriageState
):

    description = state[
        "incident_description"
    ]

    # --------------------------------------------------------
    # One Gemini call performs BOTH:
    # 1. Incident summary
    # 2. Feature extraction
    #
    # This reduces Gemini API usage.
    # --------------------------------------------------------

    prompt = f"""
You are an Enterprise ITSM Incident Triage Agent.

Analyze the following employee incident:

{description}

Perform TWO tasks:

1. Create a concise incident summary.
2. Extract the required ML features.

IMPORTANT:

For contact_type, location, u_symptom, and notify,
you MUST choose a value EXACTLY from the valid lists below.

Do NOT invent new values.

Valid contact_type values:

{feature_values['contact_type']}

Valid location values:

{feature_values['location']}

Valid u_symptom values:

{feature_values['u_symptom']}

Valid notify values:

{feature_values['notify']}


Required fields:

summary
contact_type
location
u_symptom
impact
urgency
knowledge
notify
opened_hour
opened_day_of_week
opened_month
is_weekend


Rules:

impact must be exactly one of:

"1 - High"
"2 - Medium"
"3 - Low"


urgency must be exactly one of:

"1 - High"
"2 - Medium"
"3 - Low"


knowledge must be true or false.

opened_hour must be an integer from 0 to 23.

opened_day_of_week must be an integer from 0 to 6.

opened_month must be an integer from 1 to 12.

is_weekend must be true or false.


If location or symptom is not clearly mentioned,
choose the closest reasonable value from the valid list.

Do NOT predict category.

Do NOT predict priority.

Do NOT predict SLA.


Return ONLY valid JSON.

Use exactly this structure:

{{
    "summary": "short incident summary",

    "features": {{
        "contact_type": "...",
        "location": "...",
        "u_symptom": "...",
        "impact": "...",
        "urgency": "...",
        "knowledge": true,
        "notify": "...",
        "opened_hour": 0,
        "opened_day_of_week": 0,
        "opened_month": 1,
        "is_weekend": false
    }}
}}
"""

    # --------------------------------------------------------
    # Gemini call
    # --------------------------------------------------------

    response = safe_llm_invoke(prompt)

    # --------------------------------------------------------
    # Extract response text
    # --------------------------------------------------------

    if isinstance(response.content, list):

        response_text = ""

        for item in response.content:

            if isinstance(item, dict):

                response_text += item.get(
                    "text",
                    ""
                )

            else:

                response_text += str(item)

    else:

        response_text = str(
            response.content
        )


    # --------------------------------------------------------
    # Remove markdown JSON formatting
    # --------------------------------------------------------

    response_text = response_text.replace(
        "```json",
        ""
    )

    response_text = response_text.replace(
        "```",
        ""
    )

    response_text = response_text.strip()


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            response_text
        )

    except json.JSONDecodeError:

        print(
            "\n===================================="
        )

        print(
            "WARNING: INVALID GEMINI JSON"
        )

        print(
            "===================================="
        )

        print(response_text)

        print(
            "===================================="
        )

        # Safe fallback
        result = {

            "summary":
                "Unable to generate incident summary.",

            "features": {}
        }


    # --------------------------------------------------------
    # Get summary
    # --------------------------------------------------------

    summary = result.get(
        "summary",
        ""
    )


    # --------------------------------------------------------
    # Get features
    # --------------------------------------------------------

    features = result.get(
        "features",
        {}
    )


    return {

        "summary": summary,

        "features": features
    }


# ============================================================
# 9. NORMALIZE IMPACT
# ============================================================

def normalize_impact(value):

    value = str(
        value
    ).strip().lower()


    mapping = {

        "high":
            "1 - High",

        "1 - high":
            "1 - High",

        "medium":
            "2 - Medium",

        "2 - medium":
            "2 - Medium",

        "low":
            "3 - Low",

        "3 - low":
            "3 - Low"
    }


    return mapping.get(
        value,
        "3 - Low"
    )


# ============================================================
# 10. NORMALIZE URGENCY
# ============================================================

def normalize_urgency(value):

    value = str(
        value
    ).strip().lower()


    mapping = {

        "high":
            "1 - High",

        "1 - high":
            "1 - High",

        "medium":
            "2 - Medium",

        "2 - medium":
            "2 - Medium",

        "low":
            "3 - Low",

        "3 - low":
            "3 - Low"
    }


    return mapping.get(
        value,
        "2 - Medium"
    )


# ============================================================
# 11. CONVERT BOOLEAN SAFELY
# ============================================================

def convert_to_bool(value):

    if isinstance(
        value,
        bool
    ):

        return value


    if isinstance(
        value,
        str
    ):

        value = value.strip().lower()

        if value in [
            "true",
            "1",
            "yes",
            "y",
            "on"
        ]:

            return True


        if value in [
            "false",
            "0",
            "no",
            "n",
            "off"
        ]:

            return False


    return bool(value)


# ============================================================
# 12. NODE 2
#     VALIDATE AND NORMALIZE FEATURES
# ============================================================

def validate_features(
    state: TriageState
):

    features = state[
        "features"
    ].copy()


    # --------------------------------------------------------
    # Impact
    # --------------------------------------------------------

    features["impact"] = normalize_impact(
        features.get(
            "impact",
            "3 - Low"
        )
    )


    # --------------------------------------------------------
    # Urgency
    # --------------------------------------------------------

    features["urgency"] = normalize_urgency(
        features.get(
            "urgency",
            "2 - Medium"
        )
    )


    # --------------------------------------------------------
    # Boolean values
    # --------------------------------------------------------

    features["knowledge"] = convert_to_bool(
        features.get(
            "knowledge",
            False
        )
    )


    features["is_weekend"] = convert_to_bool(
        features.get(
            "is_weekend",
            False
        )
    )


    # --------------------------------------------------------
    # Opened hour
    # --------------------------------------------------------

    try:

        features["opened_hour"] = int(
            features.get(
                "opened_hour",
                0
            )
        )

    except (
        ValueError,
        TypeError
    ):

        features["opened_hour"] = 0


    # --------------------------------------------------------
    # Day of week
    # --------------------------------------------------------

    try:

        features["opened_day_of_week"] = int(
            features.get(
                "opened_day_of_week",
                0
            )
        )

    except (
        ValueError,
        TypeError
    ):

        features["opened_day_of_week"] = 0


    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    try:

        features["opened_month"] = int(
            features.get(
                "opened_month",
                1
            )
        )

    except (
        ValueError,
        TypeError
    ):

        features["opened_month"] = 1


    # --------------------------------------------------------
    # Clamp numeric values
    # --------------------------------------------------------

    features["opened_hour"] = max(
        0,
        min(
            23,
            features["opened_hour"]
        )
    )


    features["opened_day_of_week"] = max(
        0,
        min(
            6,
            features["opened_day_of_week"]
        )
    )


    features["opened_month"] = max(
        1,
        min(
            12,
            features["opened_month"]
        )
    )


    # --------------------------------------------------------
    # Categorical values
    # --------------------------------------------------------

    features["contact_type"] = str(
        features.get(
            "contact_type",
            "Self-service"
        )
    ).strip()


    features["location"] = str(
        features.get(
            "location",
            "Unknown"
        )
    ).strip()


    features["u_symptom"] = str(
        features.get(
            "u_symptom",
            "Unknown"
        )
    ).strip()


    features["notify"] = str(
        features.get(
            "notify",
            "Do Not Notify"
        )
    ).strip()


    return {

        "features": features
    }


# ============================================================
# 13. NODE 3
#     CATEGORY PREDICTION
# ============================================================

def predict_category(
    state: TriageState
):

    features = state[
        "features"
    ]


    df = pd.DataFrame(
        [features]
    )


    prediction = category_model.predict(
        df
    )[0]


    # --------------------------------------------------------
    # Convert encoded prediction back to original label.
    #
    # Example:
    # 45 -> "Category 45"
    # --------------------------------------------------------

    predicted_category = (
        category_encoder
        .inverse_transform(
            [prediction]
        )[0]
    )


    return {

        "category":
            str(predicted_category)
    }


# ============================================================
# 14. NODE 4
#     PRIORITY PREDICTION
# ============================================================

def predict_priority(
    state: TriageState
):

    features = state[
        "features"
    ]


    df = pd.DataFrame(
        [features]
    )


    prediction = priority_model.predict(
        df
    )[0]


    predicted_priority = (
        priority_encoder
        .inverse_transform(
            [prediction]
        )[0]
    )


    return {

        "priority":
            str(predicted_priority)
    }


# ============================================================
# 15. NODE 5
#     SLA PREDICTION
# ============================================================

def predict_sla(
    state: TriageState
):

    features = state[
        "features"
    ]


    df = pd.DataFrame(
        [features]
    )


    prediction = sla_model.predict(
        df
    )[0]


    predicted_sla = convert_to_bool(
        prediction
    )


    return {

        "sla":
            predicted_sla
    }


# ============================================================
# 16. NODE 6
#     CREATE FINAL RESULT
# ============================================================

def create_final_result(
    state: TriageState
):

    final_result = {

        "incident_description":
            state[
                "incident_description"
            ],

        "summary":
            state[
                "summary"
            ],

        "validated_features":
            state[
                "features"
            ],

        "predicted_category":
            state[
                "category"
            ],

        "predicted_priority":
            state[
                "priority"
            ],

        "predicted_sla":
            state[
                "sla"
            ]
    }


    return {

        "final_result":
            final_result
    }


# ============================================================
# 17. CREATE LANGGRAPH
# ============================================================

graph = StateGraph(
    TriageState
)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

graph.add_node(
    "analyze_and_extract",
    analyze_and_extract
)


graph.add_node(
    "validate_features",
    validate_features
)


graph.add_node(
    "predict_category",
    predict_category
)


graph.add_node(
    "predict_priority",
    predict_priority
)


graph.add_node(
    "predict_sla",
    predict_sla
)


graph.add_node(
    "create_final_result",
    create_final_result
)


# ------------------------------------------------------------
# Connect nodes
# ------------------------------------------------------------

graph.add_edge(
    START,
    "analyze_and_extract"
)


graph.add_edge(
    "analyze_and_extract",
    "validate_features"
)


graph.add_edge(
    "validate_features",
    "predict_category"
)


graph.add_edge(
    "predict_category",
    "predict_priority"
)


graph.add_edge(
    "predict_priority",
    "predict_sla"
)


graph.add_edge(
    "predict_sla",
    "create_final_result"
)


graph.add_edge(
    "create_final_result",
    END
)


# ============================================================
# 18. COMPILE AGENT
# ============================================================

triage_agent = graph.compile()


# ============================================================
# 19. TEST AGENT
# ============================================================

if __name__ == "__main__":

    test_incident = (
        "Employee cannot connect to the company VPN "
        "while working from the office."
    )


    initial_state = {

        "incident_description":
            test_incident,

        "summary":
            "",

        "features":
            {},

        "category":
            "",

        "priority":
            "",

        "sla":
            False,

        "final_result":
            {}
    }


    print(
        "\n===================================="
    )

    print(
        "ENTERPRISE ITSM AI TRIAGE AGENT"
    )

    print(
        "===================================="
    )


    try:

        result = triage_agent.invoke(
            initial_state
        )


        # ----------------------------------------------------
        # Incident
        # ----------------------------------------------------

        print(
            "\nIncident:"
        )

        print(
            result[
                "incident_description"
            ]
        )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\nSummary:"
        )

        print(
            result[
                "summary"
            ]
        )


        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        print(
            "\nValidated Features:"
        )

        print(
            json.dumps(
                result[
                    "features"
                ],
                indent=2
            )
        )


        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        print(
            "\nPredicted Category:"
        )

        print(
            result[
                "category"
            ]
        )


        # ----------------------------------------------------
        # Priority
        # ----------------------------------------------------

        print(
            "\nPredicted Priority:"
        )

        print(
            result[
                "priority"
            ]
        )


        # ----------------------------------------------------
        # SLA
        # ----------------------------------------------------

        print(
            "\nPredicted SLA:"
        )

        print(
            result[
                "sla"
            ]
        )


        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        print(
            "\nFinal Result:"
        )

        print(
            json.dumps(
                result.get(
                    "final_result",
                    {}
                ),
                indent=2
            )
        )


        print(
            "\n===================================="
        )

        print(
            "TRIAGE COMPLETED SUCCESSFULLY"
        )

        print(
            "===================================="
        )


    except Exception as e:

        print(
            "\n===================================="
        )

        print(
            "TRIAGE FAILED"
        )

        print(
            "===================================="
        )

        print(
            str(e)
        )

        print(
            "===================================="
        )