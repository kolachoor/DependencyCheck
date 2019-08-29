# Databricks notebook source
dbutils.library.installPyPI("mlflow")
// logic

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Load wine dataset

# COMMAND ----------

df = spark.read.option("header", "true").option("inferSchema", True).option("sep", ";").csv("dbfs:/tmp/wine-quality.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Train local model using scikit-learn

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
  rmse = np.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  return rmse, mae, r2

# COMMAND ----------

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

np.random.seed(40)

# Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
data = df.toPandas().sample(frac=0.5)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data, test_size=0.25)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# COMMAND ----------

ALPHA = 2
L1_RATIO = 0.5

import mlflow
import mlflow.sklearn

os.system("")
with mlflow.start_run(run_name="sklearn-a%s-l%s" % (ALPHA, L1_RATIO)):
  lr = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, random_state=42)
  lr.fit(train_x, train_y)

  predicted_qualities = lr.predict(test_x)

  (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

  print("Elasticnet model (alpha=%f, l1_ratio=%f): rmse=%s" % (ALPHA, L1_RATIO, rmse))
  mlflow.log_param("alpha", ALPHA)
  mlflow.log_param("l1_ratio", L1_RATIO)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  mlflow.sklearn.log_model(lr, ".")

# COMMAND ----------

# MAGIC %md ## Part 3: Train distributed model using SparkML

# COMMAND ----------

ALPHA = 2
L1_RATIO = 0.5

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# Assemble feature columns into a vector (excluding the "quality" label)
assembler = VectorAssembler(
    inputCols=df.columns[0:-1],
    outputCol="features")

df_features = assembler.transform(df)

# Split into a training and test dataset.
(training, test) = df_features.randomSplit([0.8, 0.2])

with mlflow.start_run(run_name="spark-a%s-l%s" % (ALPHA, L1_RATIO)):
  lr = LinearRegression(maxIter=10, regParam=ALPHA, elasticNetParam=L1_RATIO, labelCol="quality")

  lrModel = lr.fit(training)
  print("Coefficients: %s" % str(lrModel.coefficients))
  print("Intercept: %s" % str(lrModel.intercept))
  
  predictions = lrModel.transform(test)
  pdf = predictions.toPandas()
  (rmse, mae, r2) = eval_metrics(pdf['quality'], pdf['prediction'])
  print("RMSE: %f" % rmse)

  mlflow.log_param("alpha", ALPHA)
  mlflow.log_param("l1_ratio", L1_RATIO)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)