# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Serving: Loading the Registered Model
# MAGIC
# MAGIC Demonstrates pulling the model **by registry name and stage**, not by a
# MAGIC hardcoded run ID — this is what makes the model swappable in production
# MAGIC without touching the app code that consumes it (e.g. your Streamlit app).

# COMMAND ----------

import mlflow
import pandas as pd

dbutils.widgets.text("schema", "churn_project", "Schema")
schema = dbutils.widgets.get("schema")
model_name = f"{schema}_churn_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote the latest version to "Production" (optional, one-time)
# MAGIC
# MAGIC In a real workflow this happens after a human or CI check reviews the
# MAGIC candidate model's metrics — not automatically on every training run.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Promoted {model_name} v{latest_version} to Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load by stage and predict
# MAGIC
# MAGIC This is the call your Streamlit app (or any external service) would make —
# MAGIC it never needs to know a specific run ID, only "give me whatever is in
# MAGIC Production right now."

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# Example: predict on a small batch pulled from the silver table
sample = spark.table(f"hive_metastore.{schema}.silver_churn").limit(5).toPandas()
feature_cols = [c for c in sample.columns if c not in ("customerID", "Churn", "Churn_Label")]

predictions = model.predict(sample[feature_cols])
sample["predicted_churn"] = predictions
display(sample[["customerID", "Churn", "predicted_churn"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wiring this into Streamlit
# MAGIC
# MAGIC Your existing Streamlit app can load the same model with:
# MAGIC ```python
# MAGIC import mlflow
# MAGIC model = mlflow.pyfunc.load_model("models:/churn_project_churn_model/Production")
# MAGIC ```
# MAGIC as long as it can reach the Databricks tracking server (via `MLFLOW_TRACKING_URI`
# MAGIC and a Databricks personal access token). This is the "one sentence" answer if
# MAGIC an interviewer asks how the Databricks side connects back to your existing app.
