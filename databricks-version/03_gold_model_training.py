# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold Layer: Model Training with MLflow
# MAGIC
# MAGIC Trains Logistic Regression and XGBoost on the silver table, with
# MAGIC **MLflow** tracking every run (params, metrics, model artifact, SHAP plot),
# MAGIC and registers the best model in the **MLflow Model Registry**.
# MAGIC
# MAGIC This notebook is the single biggest signal in the whole project: it shows
# MAGIC you can operate the part of the platform a Databricks partner actually sells.

# COMMAND ----------

# MAGIC %pip install xgboost shap imbalanced-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

dbutils.widgets.text("catalog", "hive_metastore", "Catalog")
dbutils.widgets.text("schema", "churn_project", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
silver_table = f"{catalog}.{schema}.silver_churn"

mlflow.set_experiment(f"/Shared/churn_prediction_{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pull silver table into pandas for sklearn/XGBoost
# MAGIC
# MAGIC At this dataset size a pandas conversion is fine and matches your original
# MAGIC pipeline's modeling code almost 1:1. For genuinely large silver tables,
# MAGIC the same logic would move to `pyspark.ml` — worth mentioning if asked
# MAGIC "what would you change at 10x the data."

# COMMAND ----------

pdf = spark.table(silver_table).toPandas()

target_col = "Churn_Label"
drop_cols = ["customerID", "Churn", target_col]
categorical_cols = [c for c in pdf.select_dtypes(include="object").columns if c not in drop_cols]
numeric_cols = [c for c in pdf.select_dtypes(include=np.number).columns if c not in drop_cols]

X = pdf[numeric_cols + categorical_cols]
y = pdf[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Churn rate: {y.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing pipeline + SMOTE-after-split
# MAGIC
# MAGIC SMOTE is applied only to the **training** fold, after the split — the same
# MAGIC leakage-avoidance decision from the original project, preserved here.

# COMMAND ----------

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
print(f"After SMOTE — class balance: {np.bincount(y_train_res)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run 1 — Logistic Regression (MLflow-tracked)

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression") as run:
    params = {"C": 1.0, "max_iter": 1000, "class_weight": None}
    mlflow.log_params(params)

    lr = LogisticRegression(**params)
    lr.fit(X_train_res, y_train_res)

    preds = lr.predict(X_test_proc)
    proba = lr.predict_proba(X_test_proc)[:, 1]

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(lr, "model")

    print(f"Logistic Regression — F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    lr_run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run 2 — XGBoost with GridSearchCV (MLflow-tracked)

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_gridsearch") as run:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }

    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    grid = GridSearchCV(xgb, param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test_proc)
    proba = best_model.predict_proba(X_test_proc)[:, 1]

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.xgboost.log_model(best_model, "model")

    print(f"XGBoost (best) — F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    print(f"Best params: {grid.best_params_}")
    xgb_run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP explainability — logged as an MLflow artifact
# MAGIC
# MAGIC Same explainability step as the original project, but now the plot lives
# MAGIC attached to the run instead of as a standalone notebook cell output.

# COMMAND ----------

with mlflow.start_run(run_id=xgb_run_id):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_proc[:200])  # sample for speed

    feature_names = (
        numeric_cols +
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    )

    plt.figure()
    shap.summary_plot(shap_values, X_test_proc[:200], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("/tmp/shap_summary.png")
    mlflow.log_artifact("/tmp/shap_summary.png")
    plt.close()

print("SHAP summary plot logged to the XGBoost run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the best model in the MLflow Model Registry
# MAGIC
# MAGIC This is the step that turns "I trained a model" into "I shipped a model
# MAGIC a real serving layer or Streamlit app can pull by name/version" —
# MAGIC directly relevant to a Databricks partner's day-to-day work.

# COMMAND ----------

model_name = f"{schema}_churn_model"
model_uri = f"runs:/{xgb_run_id}/model"

registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Registered model: {registered_model.name}, version: {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare runs
# MAGIC
# MAGIC Query the MLflow experiment programmatically — useful to show in an
# MAGIC interview screen-share instead of scrolling the UI.

# COMMAND ----------

runs_df = mlflow.search_runs(order_by=["metrics.f1_score DESC"])
display(runs_df[["run_id", "tags.mlflow.runName", "metrics.f1_score", "metrics.roc_auc"]])
