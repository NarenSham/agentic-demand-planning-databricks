# notebooks/Utils/governance_logging.py
# Governance logging utility — dual-mode.
#
# Notebook / Lakeflow job context: a Spark session is attached, writes go
# through Spark exactly as before.
#
# Model Serving / Databricks App context: no Spark session exists. Writes
# go through the SQL warehouse via databricks-sql-connector instead, using
# the service principal credentials Databricks injects into that runtime.
#
# Both paths write the same two tables with the same schema, so callers
# never need to know which mode is active.

import os
import uuid
from datetime import datetime, timezone


def _get_spark():
    """
    Returns an active SparkSession if one exists, None otherwise.
    Never raises — absence of Spark is expected in Serving/App context,
    not an error.
    """
    try:
        from pyspark.sql import SparkSession
        # getActiveSession does NOT create a new session — it returns
        # None if nothing is attached, which is what we want to detect.
        return SparkSession.getActiveSession()
    except Exception:
        return None


def _sql_warehouse_connect():
    """
    Opens a connection to the SQL warehouse for the governance tables.

    Reuses databricks.sdk.core.Config for auth instead of hand-rolled
    host/token env vars — this is the same automatic-authentication
    mechanism the Model Serving container already relies on successfully
    elsewhere in this codebase (see WorkspaceClient() in
    15_build_demand_agent.py, used there to reach the LLM serving
    endpoint). Reusing it here means one less untested assumption about
    what credentials the serving container actually has.

    Only needs ONE piece of config beyond that: the target warehouse's
    HTTP path, since a warehouse connection is warehouse-specific and
    can't be inferred from workspace auth alone. Set via
    GOVERNANCE_WAREHOUSE_HTTP_PATH (Apps: app.yaml env block; notebook/
    job context never reaches this path since Spark is used instead).

    Returns None if the http path isn't configured or auth fails —
    callers treat that as "logging unavailable" and no-op.
    """
    http_path = os.environ.get("GOVERNANCE_WAREHOUSE_HTTP_PATH")
    if not http_path:
        return None

    try:
        from databricks.sdk.core import Config
        from databricks import sql as databricks_sql

        cfg = Config()  # resolves the same auto-injected credentials
                         # WorkspaceClient() already uses in this container
        return databricks_sql.connect(
            server_hostname      = cfg.host.replace("https://", ""),
            http_path            = http_path,
            credentials_provider = lambda: cfg.authenticate,
        )
    except Exception:
        return None


def _sql_warehouse_insert(table: str, columns: list, values: tuple):
    """
    Inserts one row via the SQL warehouse. Silent no-op on any failure —
    governance logging must never take down the caller (agent or app).
    """
    conn = _sql_warehouse_connect()
    if conn is None:
        return False

    try:
        placeholders = ", ".join(["?"] * len(columns))
        col_list     = ", ".join(columns)
        with conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
                values,
            )
        return True
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def log_decision(
    agent_name: str,
    action: str,
    details: str,
    model_version: str = None
):
    """
    Log a decision to the governance audit table.
    Call this from any notebook, job, served agent, or App for any
    significant event. Works identically regardless of runtime context.
    """
    decision_id = str(uuid.uuid4())
    logged_at   = datetime.now(timezone.utc)

    spark = _get_spark()
    if spark is not None:
        from pyspark.sql import Row
        row = Row(
            decision_id = decision_id,
            logged_at   = logged_at,
            agent_name  = agent_name,
            action      = action,
            details     = details,
        )
        df = spark.createDataFrame([row])
        (df.write
            .format("delta")
            .mode("append")
            .saveAsTable("cpg_planning.governance.ai_decision_log"))
        return

    # No Spark session — fall back to SQL warehouse (Serving / App context)
    _sql_warehouse_insert(
        table   = "cpg_planning.governance.ai_decision_log",
        columns = ["decision_id", "logged_at", "agent_name", "action", "details"],
        values  = (decision_id, logged_at, agent_name, action, details),
    )


def log_tool_call(
    agent_name: str,
    tool_name: str,
    parameters: str,
    result_summary: str
):
    """
    Log a tool call to the tool call history table.
    Same dual-mode behavior as log_decision.
    """
    call_id   = str(uuid.uuid4())
    called_at = datetime.now(timezone.utc)

    spark = _get_spark()
    if spark is not None:
        from pyspark.sql import Row
        row = Row(
            call_id        = call_id,
            called_at      = called_at,
            agent_name     = agent_name,
            tool_name      = tool_name,
            parameters     = parameters,
            result_summary = result_summary,
        )
        df = spark.createDataFrame([row])
        (df.write
            .format("delta")
            .mode("append")
            .saveAsTable("cpg_planning.governance.tool_call_history"))
        return

    _sql_warehouse_insert(
        table   = "cpg_planning.governance.tool_call_history",
        columns = ["call_id", "called_at", "agent_name", "tool_name", "parameters", "result_summary"],
        values  = (call_id, called_at, agent_name, tool_name, parameters, result_summary),
    )
