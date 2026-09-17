# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze Layer: Raw Ingestion
# MAGIC
# MAGIC Ingests the raw Telco Customer Churn CSV into a Delta Lake **bronze** table.
# MAGIC The bronze layer stores data as close to its raw form as possible — no cleaning,
# MAGIC no type-casting, just a durable, versioned, queryable copy of the source file.
# MAGIC
# MAGIC This is the step that turns "I had a CSV" into "I built a lakehouse pipeline."

# COMMAND ----------

# MAGIC %md
# MAGIC ### Config

# COMMAND ----------

dbutils.widgets.text("catalog", "hive_metastore", "Catalog")
dbutils.widgets.text("schema", "churn_project", "Schema")
dbutils.widgets.text("raw_path", "/FileStore/tables/telco_churn.csv", "Raw CSV path")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_path = dbutils.widgets.get("raw_path")

bronze_table = f"{catalog}.{schema}.bronze_churn"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload note
# MAGIC If you're running this in Databricks Community Edition:
# MAGIC 1. Go to **Data > Add Data > Upload File** and upload `WA_Fn-UseC_-Telco-Customer-Churn.csv`
# MAGIC 2. Note the DBFS path it gives you (usually `/FileStore/tables/<filename>`)
# MAGIC 3. Update the `raw_path` widget above to match

# COMMAND ----------

df_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)  # bronze = keep everything as string; cast happens in silver
    .csv(raw_path)
)

print(f"Rows ingested: {df_raw.count()}")
display(df_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta (bronze)
# MAGIC
# MAGIC Delta gives us ACID writes + time travel — if a future ingestion run is bad,
# MAGIC we can roll the table back to a previous version. That's a real answer to
# MAGIC "why Delta over just Parquet or CSV" if it comes up in an interview.

# COMMAND ----------

(
    df_raw.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)

print(f"Bronze table written: {bronze_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity check + time travel demo

# COMMAND ----------

spark.sql(f"DESCRIBE HISTORY {bronze_table}").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Bronze is intentionally messy — e.g. `TotalCharges` is still a string here,
# MAGIC including the blank-string rows that caused the dtype bug in the original
# MAGIC pandas version of this project. We fix that explicitly and visibly in silver,
# MAGIC which is a better interview story than fixing it silently.
