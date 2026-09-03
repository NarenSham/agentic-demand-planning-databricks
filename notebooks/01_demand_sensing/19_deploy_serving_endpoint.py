# Databricks notebook source
# 19_deploy_serving_endpoint.py
# Deploys cpg_planning.ml.demand_agent@champion to a Model Serving endpoint.
#
# Run this AFTER 15b_log_agent.py has registered a model version that
# includes the fixed governance_logging.py (dual-mode) and the corrected
# code_paths (whole Utils/ directory, not a single flattened file).
#
# Idempotent: safe to re-run. If the endpoint exists, it updates the
# served entity to point at the current @champion version instead of
# failing on "already exists".

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

ENDPOINT_NAME     = "cpg-demand-agent"
UC_MODEL_NAME     = "cpg_planning.ml.demand_agent"
MODEL_ALIAS       = "champion"

w = WorkspaceClient()

# COMMAND ----------
# Resolve @champion to a concrete version number.
# Model Serving endpoints pin to a version, not an alias — so every time
# 15b_log_agent.py promotes a new champion, this notebook needs a re-run
# to move the endpoint. That's a deliberate, visible step, not automatic
# drift — you don't want a served agent silently changing behavior.

client        = mlflow.MlflowClient() if "mlflow" in dir() else __import__("mlflow").MlflowClient()
model_version = client.get_model_version_by_alias(UC_MODEL_NAME, MODEL_ALIAS)

print(f"Deploying {UC_MODEL_NAME} version {model_version.version} "
      f"(champion) to endpoint '{ENDPOINT_NAME}'")

# COMMAND ----------
# Endpoint config.
# scale_to_zero_enabled=True: this is a portfolio project, not production
# traffic. Cost when idle should be ~zero. Tradeoff is a cold-start delay
# (seconds, sometimes closer to a minute) on the first request after idle
# — the App's chat UI needs to show that honestly (spinner + copy that
# says "waking up the model", not a bare hang) rather than pretending
# every response is instant.

served_entity = ServedEntityInput(
    entity_name                = UC_MODEL_NAME,
    entity_version              = model_version.version,
    workload_size               = "Small",
    scale_to_zero_enabled       = True,
)

endpoint_config = EndpointCoreConfigInput(served_entities=[served_entity])

# COMMAND ----------
# Create or update.

existing = None
try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
except Exception:
    existing = None

if existing is None:
    print(f"Creating new endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(
        name   = ENDPOINT_NAME,
        config = endpoint_config,
    )
    print("Endpoint created.")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' exists — updating served entity "
          f"to version {model_version.version}...")
    w.serving_endpoints.update_config_and_wait(
        name            = ENDPOINT_NAME,
        served_entities = [served_entity],
    )
    print("Endpoint updated.")

# COMMAND ----------
# Smoke test — same request shape 15_build_demand_agent.py's test_agent()
# uses, sent over the wire instead of in-process. Confirms the served
# agent actually answers, not just that the endpoint came up healthy.

import json
from uuid import uuid4

response = w.serving_endpoints.query(
    name   = ENDPOINT_NAME,
    inputs = {
        "input": [
            {"role": "user", "content": "How current is the demand data?"}
        ],
        "custom_inputs": {"session_id": str(uuid4())}
    },
)

print("Smoke test response:")
print(json.dumps(response.as_dict(), indent=2)[:1500])

# COMMAND ----------
# Grant the App's service principal permission to query this endpoint.
# Fill in the App's service principal application ID after the App is
# created in the workspace UI (Compute > Apps > <app name> > Settings) —
# app.py won't be able to call this endpoint without this grant.

# from databricks.sdk.service.serving import ServingEndpointPermissionLevel
#
# w.serving_endpoints.set_permissions(
#     serving_endpoint_id = w.serving_endpoints.get(ENDPOINT_NAME).id,
#     access_control_list = [{
#         "service_principal_name": "<APP_SERVICE_PRINCIPAL_APPLICATION_ID>",
#         "permission_level": "CAN_QUERY",
#     }],
# )
