import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

incident_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "technical_problem_type": {"type": "string"},
        "application_or_service": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"}
        },
        "suggested_impact": {
            "type": "string",
            "enum": ["1 - High", "2 - Medium", "3 - Low"]
        },
        "suggested_urgency": {
            "type": "string",
            "enum": ["1 - High", "2 - Medium", "3 - Low"]
        }
    },
    "required": [
        "summary",
        "technical_problem_type",
        "application_or_service",
        "keywords",
        "suggested_impact",
        "suggested_urgency"
    ]
}


def analyze_incident(description: str):

    prompt = f"""
You are an AI assistant for an Enterprise IT Service Management system.

Analyze this employee incident:

"{description}"

Provide a summary, the technical problem type, the application or service
involved, important keywords, and the suggested impact and urgency.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": incident_schema
        }
    )

    return json.loads(response.text)