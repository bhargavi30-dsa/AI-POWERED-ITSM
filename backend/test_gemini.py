from gemini_service import analyze_incident


description = """
I cannot access the payroll application.
When I try to log in, I receive an authentication error.
"""


result = analyze_incident(description)

print("\nGemini Response:")
print(result)