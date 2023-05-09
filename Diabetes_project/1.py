# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 19:58:57 2023

@author: User
"""
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from pyspark.ml import Pipeline

# Importing the data
covid_data_path = "/content/covid_data.csv"

spark = SparkSession.builder.appName('COVID19 Clustering').getOrCreate()

# load the COVID-19 dataset as a Spark dataframe
df = spark.read.csv('covid_data.csv', header=True, inferSchema=True)
df.show()

# columns_to_drop = ["backupnotes", "contractedfromwhichpatientsuspected", "detectedcity","detecteddistrict",""]
# df = df.drop(*columns_to_drop)
df_selected = df.select('agebracket', 'currentstatus', 'dateannounced', 'detectedstate', 'gender', 'nationality', 'statuschangedate')
df_selected.show()

from pyspark.sql.functions import to_date, col, datediff
df_selected = df_selected.dropna(how="any")
# df = df.filter(df.currentstatus != NAN)
# df_selected = df_selected.withColumn("dateannounced", to_date("dateannounced", "dd/MM/yyyy"))
# df_selected = df_selected.withColumn("statuschangedate", to_date("statuschangedate", "dd/MM/yyyy"))
# df_selected = df_selected.withColumn('days_to_recovery', datediff(col('statuschangedate'), col('dateannounced')))

df_selected = df_selected.select('agebracket', 'detectedstate', 'gender', 'nationality', 'currentstatus',
               to_date(col('dateannounced'), 'dd/MM/yyyy').alias('dateannounced'),
               to_date(col('statuschangedate'), 'dd/MM/yyyy').alias('statuschangedate'))

# Create a new column with the difference in days between statuschangedate and dateannounced
df_selected = df_selected.withColumn('days_to_recovery', datediff(col('statuschangedate'), col('dateannounced')))

df_selected.describe().show()
df_selected.show()


y = df_selected.select("currentstatus")
x = df_selected.select("agebracket", "detectedstate", "gender", "nationality", "days_to_recovery")
x.show()

# Convert categorical columns into numeric form
indexers = [
    StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep")
    for column in ["agebracket","detectedstate", "gender", "nationality"]
]

# Create vector of selected columns
assembler = VectorAssembler(inputCols=["agebracket_index", "detectedstate_index", "gender_index", "nationality_index", "days_to_recovery"],
                            outputCol="features")

# Fit k-means model
kmeans = KMeans(k=2, seed=1)
pipeline = Pipeline(stages=indexers + [assembler, kmeans])
model = pipeline.fit(x)

# Get predictions
predictions = model.transform(x)
predictions.show()

predictions.select("prediction").show()
y.show()