# 15b_log_agent.py
import sys
import mlflow
import subprocess
mlflow.set_registry_uri("databricks-uc")
username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(
    f"/Users/{username}/Databricks-CPG/experiments/demand_sensing"
)
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "backoff"])

# ── Input example ─────────────────────────────────────────────────────────────
# Must match the exact shape Model Serving will POST to the agent.
# ResponsesAgent expects: {input: [...], custom_inputs: {...}}
# custom_inputs carries session_id — required for memory continuity.

input_example = {
    "input": [
        {"role": "user", "content": "What are forecasted Food sales in Ontario next month?"}
    ],
    "custom_inputs": {
        "session_id": "example-session-001"
    }
}

# ── Signature ─────────────────────────────────────────────────────────────────
# ResponsesAgent has a standard signature — use the MLflow helper.
# Do NOT hand-roll Schema here; infer_signature from the pyfunc type
# guarantees it matches what Model Serving validates against.

signature = mlflow.models.infer_signature(
    model_input  = input_example,
    model_output = {
        "output": [
            {"type": "message", "content": [{"type": "text", "text": "Forecasted retail sales: $2.3B"}]}
        ],
        "custom_outputs": {"session_id": "example-session-001"}
    }
)

# Step 1 — Log
with mlflow.start_run(run_name="demand_agent_v1") as run:
    logged = mlflow.pyfunc.log_model(
        name             = "demand_agent",
        python_model     = f"/Workspace/Users/{username}/Databricks-CPG/notebooks/01_demand_sensing/15_build_demand_agent.py",
        code_paths       = [
            # Whole directory, not the single file — preserves the Utils/
            # package structure so `from Utils.governance_logging import
            # log_decision` resolves inside the Serving container the same
            # way it does in the notebook. A single-file path here gets
            # flattened by MLflow and silently breaks that import.
            f"/Workspace/Users/{username}/Databricks-CPG/notebooks/Utils"
        ],
        pip_requirements = [
            "xgboost", "mlflow", "databricks-sdk", "backoff",
            "databricks-sql-connector",  # governance logging fallback path
        ],
        input_example    = input_example,
        signature        = signature,
    )
    print(f"Logged: {logged.model_uri}")

# Step 2 — Register
registered = mlflow.register_model(
    model_uri = logged.model_uri,
    name      = "cpg_planning.ml.demand_agent"
)
print(f"Registered: v{registered.version}")

# Step 3 — Promote
client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name    = "cpg_planning.ml.demand_agent",
    alias   = "champion",
    version = registered.version
)
print(f"Champion alias set to v{registered.version}")