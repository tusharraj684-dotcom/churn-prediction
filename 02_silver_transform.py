# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Layer: Cleaning & Feature Engineering
# MAGIC
# MAGIC Reads the bronze table, fixes the known data-quality issue (the
# MAGIC `TotalCharges` blank-string bug from the original project), casts types,
# MAGIC and does feature engineering — all in PySpark instead of pandas so it
# MAGIC scales past a single machine.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

dbutils.widgets.text("catalog", "hive_metastore", "Catalog")
dbutils.widgets.text("schema", "churn_project", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

bronze_table = f"{catalog}.{schema}.bronze_churn"
silver_table = f"{catalog}.{schema}.silver_churn"

# COMMAND ----------

df = spark.table(bronze_table)
print(f"Bronze rows: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The TotalCharges bug, fixed explicitly
# MAGIC
# MAGIC In the original pandas version, `TotalCharges` silently loaded as an
# MAGIC `object` dtype because ~11 rows contain a blank string instead of a
# MAGIC number (new customers with 0 tenure). Here we surface that count
# MAGIC explicitly — good practice, and a great thing to point at in an interview.

# COMMAND ----------

blank_count = df.filter(F.trim(F.col("TotalCharges")) == "").count()
print(f"Blank TotalCharges rows found: {blank_count}")

df_clean = (
    df
    .withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges")) == "", None)
         .otherwise(F.col("TotalCharges"))
         .cast(DoubleType())
    )
    .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast(DoubleType()))
    .withColumn("tenure", F.col("tenure").cast("int"))
)

# Fill the now-null TotalCharges with 0 — these are all tenure==0 new customers
df_clean = df_clean.fillna({"TotalCharges": 0.0})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering
# MAGIC
# MAGIC Recreating the feature set from the pandas pipeline, plus a couple of
# MAGIC lakehouse-scale-friendly additions.

# COMMAND ----------

df_feat = (
    df_clean
    .withColumn("Churn_Label", F.when(F.col("Churn") == "Yes", 1).otherwise(0))
    .withColumn(
        "AvgMonthlySpend",
        F.when(F.col("tenure") > 0, F.col("TotalCharges") / F.col("tenure"))
         .otherwise(F.col("MonthlyCharges"))
    )
    .withColumn(
        "TenureBucket",
        F.when(F.col("tenure") <= 12, "0-1yr")
         .when(F.col("tenure") <= 24, "1-2yr")
         .when(F.col("tenure") <= 48, "2-4yr")
         .otherwise("4yr+")
    )
    .withColumn(
        "HasMultipleServices",
        (
            (F.col("OnlineSecurity") == "Yes").cast("int") +
            (F.col("OnlineBackup") == "Yes").cast("int") +
            (F.col("DeviceProtection") == "Yes").cast("int") +
            (F.col("TechSupport") == "Yes").cast("int")
        )
    )
)

display(df_feat.select(
    "customerID", "tenure", "TotalCharges", "AvgMonthlySpend",
    "TenureBucket", "HasMultipleServices", "Churn_Label"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta (silver)

# COMMAND ----------

(
    df_feat.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)

print(f"Silver table written: {silver_table}")
print(f"Silver rows: {df_feat.count()}, columns: {len(df_feat.columns)}")
