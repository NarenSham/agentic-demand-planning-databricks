# Agentic Demand Planning on Databricks

A compound AI system for CPG retail demand forecasting built on Databricks, grounded in Statistics Canada retail trade data, and designed to be *interrogated*, not just trusted.

Most forecasting tools give you a number. This gives you a number, a confidence interval, the reasoning behind it, and an agent you can ask "why?" — with every answer traced back to real feature importances and a documented model, not a hallucinated explanation.

> **First public forecast shipped July 2026.** Actuals publish via Statistics Canada in August 2026 — this system's track record is verifiable, not asserted. See [`predictions/`](./predictions).

---

## Why this exists

CPG demand planning teams typically choose between two bad options: a black-box forecasting tool nobody trusts enough to act on, or a spreadsheet model that doesn't scale past one analyst's head. This project is an attempt at a third option — a forecasting system architected so that *the reasoning is as inspectable as the number*, built the way I'd architect it for an enterprise client: governed, evaluated, and honest about its own limitations.

It's also a working answer to a specific architecture question: **what does "agentic" actually buy you over a scheduled ML job?** The answer here is a system a planner can question in natural language — "why did Ontario General Merchandise get a wider confidence band than Quebec?" — and get a grounded, SHAP-backed answer rather than a plausible-sounding guess.

---

## Architecture

**Medallion lakehouse**, six governed schemas in Unity Catalog (`cpg_planning`):

| Layer | Purpose |
|---|---|
| `bronze` | Source-faithful StatCan retail data + external signals, exactly as pulled |
| `silver` | Cleaned, quality-gated retail trade (Lakeflow Declarative Pipeline constraints — bad dates, unknown geographies, and invalid values are dropped or fail the pipeline) |
| `gold` | Feature table (lags, rolling averages) + `demand_predictions`, the public track record table |
| `ml` | Registered UC functions (agent tools), model artifacts |
| `governance` | Full audit trail — every agent decision and tool call logged |
| `monitoring` | Model drift and retraining triggers |

**Orchestration:** a Databricks Asset Bundle (`databricks.yml`) deploys a monthly Lakeflow job that runs the full pipeline end-to-end — ingestion through prediction — on a schedule, with failure alerting.

---

## What's built

| Module | What it does |
|---|---|
| **Data collection** (`data_collection/`) | Pulls StatCan retail trade, CPI, gas prices, Google Trends |
| **Medallion pipeline** (`notebooks/01_demand_sensing/01–11`) | Bronze → silver → gold, feature engineering (lags, rolling windows), quality-gated via Declarative Pipelines (`pipeline/`) |
| **Forecasting model** | Pooled XGBoost across 5 categories × 6 provinces — **4.68% MAPE**, time-series holdout (last 12 months, no random split to avoid leakage) |
| **Explainability** (`13_shap_importance`) | SHAP feature importance per category, feeding the agent's `explain_prediction` tool |
| **Agent tools** (`14_create_agent_tools`) | Unity Catalog SQL functions for data-freshness and SHAP-based explanation queries |
| **Compound AI agent** (`15_build_demand_agent`) | MLflow `ResponsesAgent` — natural-language interface over the forecasting system, with governance logging on every call |
| **Eval harness** (`16_evaluate_agent`) | Three tiers: deterministic guardrail checks → forecast-range validation (±15% tolerance against actuals) → LLM-judged reasoning quality via `mlflow.genai.evaluate` |
| **Backtest** (`17_backtest`) | 14-month rolling backtest, Jan 2025–Feb 2026, with an explicit honest note on which months are genuinely out-of-sample |
| **Monthly orchestration** (`18_lakeflow_job`) | End-to-end pipeline run, conditional retraining (only fires if recent MAPE degrades >20% vs. training MAPE) |

---

## Engineering decisions worth reading

The full reasoning — including tradeoffs I *didn't* take — is in [`docs/model_decisions.md`](./docs/model_decisions.md). A few worth highlighting:

- **Pooled model over per-category models.** 2,940 training rows total; per-category models would train on ~200 rows each. Pooled model wins on generalization, at the cost of pulling Clothing (8.6% MAPE) and Gasoline toward the pooled average. Documented, not hidden.
- **External signals tested and rejected.** CPI, gas prices, and Google Trends all made the model *worse* (5.83% MAPE vs. 4.68% lag-only) at this aggregation level. Kept the negative result in the record — it's a real finding, not a failed experiment to bury.
- **Python tools over UC SQL functions for compute-heavy agent actions.** SHAP-to-language translation and MLflow model inference can't cleanly live inside a UC SQL function sandbox. UC functions stay as the documented data contract; compute-heavy tools live in Python.
- **Multi-agent architecture is scoped, not hand-waved.** The roadmap specifies specialist agents (demand, price, promo) as independent systems connected via an MCP orchestrator using `UCFunctionToolkit` for discovery — with a stated build order, not just an aspiration.

---

## Current results

- **4.68% MAPE** (pooled model, time-series holdout)
- **4.85% backtest MAPE** on the first public forecast (July 2026), against 14 months of rolling backtest history
- Confidence bands vary meaningfully by category — Food & Beverage (±3.0%) vs. Clothing (±8.6%) — because the model's own uncertainty is exposed, not smoothed over

---

## Tech stack

Databricks (Lakeflow Declarative Pipelines, Unity Catalog, MLflow, Databricks Asset Bundles) · XGBoost · SHAP · MLflow `ResponsesAgent` · `mlflow.genai.evaluate` · Delta Lake · PySpark

---

## Roadmap

- **Databricks App** — planner-facing UI over the agent (next)
- **Price agent** — second specialist, after the app ships
- **MCP orchestrator** — connects demand + price (+ promo) specialists once both are independently stable and evaluated
- **Lakebase-backed session storage** — replacing the current in-memory session store once multi-session planner use is real (interface already stubbed for the swap)

---

## Repository structure

```
├── data_collection/       # StatCan, CPI, gas prices, Google Trends pulls
├── infrastructure/        # Catalog, schema, and governance table DDL
├── pipeline/               # Lakeflow Declarative Pipeline (bronze → silver, quality-gated)
├── notebooks/01_demand_sensing/   # Full build: ingestion → model → agent → eval → orchestration
├── docs/model_decisions.md        # Every architectural decision, with reasoning and tradeoffs
├── predictions/            # Published forecasts, with confidence intervals and methodology notes
└── databricks.yml          # Asset Bundle — monthly orchestration job definition
```

---

Built and maintained by [Naren Sham](https://www.linkedin.com/in/narensham/) 
