import { useState } from "react";
import "./App.css";
const API_URL = "http://127.0.0.1:8000";

function App() {
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [incidentDescription, setIncidentDescription] = useState("");
  const [result, setResult] = useState(null);

  // ---------------- TRIAGE ----------------

  const triageIncident = async (event) => {
    event.preventDefault();

    if (!incidentDescription.trim()) {
      setError("Please describe the incident before submitting.");
      return;
    }

    setPredicting(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/complete-triage`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          incident_description: incidentDescription,
        }),
      });

      if (!response.ok) {
        throw new Error("Triage request failed");
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error(err);
      setError(
        "Triage failed. Check the FastAPI terminal for the exact error."
      );
    } finally {
      setPredicting(false);
    }
  };

  // ---------------- UI ----------------

  return (
    <div className="app-container">

      <header>
        <h1>Enterprise AI ITSM</h1>
        <p>Intelligent Incident Management</p>
      </header>

      <main>

        <section className="incident-description">
          <h2>Describe Your Incident</h2>

          <p>
            Describe your IT issue in your own words. Agent 1 extracts the
            relevant details and predicts category, priority, and SLA risk.
            Agent 2 then assigns the correct team and recommends a resolver
            based on historical incident data.
          </p>

          <form onSubmit={triageIncident}>
            <textarea
              value={incidentDescription}
              onChange={(e) => setIncidentDescription(e.target.value)}
              placeholder="Example: I am unable to access the payroll application and receive an authentication error when logging in."
              maxLength={500}
              rows={5}
            />

            <div className="character-count">
              {incidentDescription.length} / 500
            </div>

            <button type="submit" disabled={predicting}>
              {predicting ? "Triaging Incident..." : "Triage Incident"}
            </button>
          </form>
        </section>

        {error && <div className="error">{error}</div>}

        <div className="grid">

          {/* ---------------- AGENT 1 RESULTS ---------------- */}

          <section className="card results-card">
            <h3>Agent 1 — Triage</h3>

            {result ? (
              <div className="results">
                <div className="result">
                  <span>Predicted Category</span>
                  <strong>{result.predicted_category || "—"}</strong>
                </div>

                <div className="result">
                  <span>Predicted Priority</span>
                  <strong>{result.predicted_priority || "—"}</strong>
                </div>

                <div className="result">
                  <span>SLA Status</span>
                  <strong
                    className={
                      result.predicted_sla ? "sla-success" : "sla-warning"
                    }
                  >
                    {result.predicted_sla ? "SLA Met" : "SLA Breach Risk"}
                  </strong>
                </div>
              </div>
            ) : (
              <div className="empty-results">
                <div className="robot">AI</div>
                <p>Submit an incident to receive AI-powered predictions.</p>
              </div>
            )}
          </section>

          {/* ---------------- AGENT 2 RESULTS ---------------- */}

          {result && (
            <section className="card results-card">
              <h3>Agent 2 — Assignment</h3>

              <div className="results">
                <div className="result">
                  <span>Assignment Group</span>
                  <strong>{result.predicted_assignment_group || "—"}</strong>
                </div>

                <div className="result">
                  <span>Recommended Resolver</span>
                  <strong>{result.recommended_resolver || "—"}</strong>
                </div>
              </div>

              {result.alternative_resolvers?.length > 0 && (
                <div className="analysis-card">
                  <h4>Alternative Resolvers</h4>
                  <div className="keywords">
                    {result.alternative_resolvers.map((resolver, index) => (
                      <span key={index} className="keyword">
                        {resolver.resolved_by} ({resolver.incident_count} incidents)
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* ---------------- AGENT ANALYSIS ---------------- */}

          {result && (
            <section className="card gemini-analysis">
              <h3>Agent Analysis</h3>

              <div className="analysis-card">
                <h4>Summary</h4>
                <p>{result.summary}</p>
              </div>

              <div className="analysis-card">
                <h4>Extracted Features</h4>
                <div className="keywords">
                  {Object.entries(result.validated_features || {}).map(
                    ([key, value]) => (
                      <span key={key} className="keyword">
                        {key}: {String(value)}
                      </span>
                    )
                  )}
                </div>
              </div>
            </section>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;