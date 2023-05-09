# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 20:06:24 2023

@author: User
"""

# Install Elasticsearch-Hadoop connector
!pip install elasticsearch==7.10.0
!pip install "pyspark==3.2.0" --ignore-installed py4j

# Set Elasticsearch cluster configuration
es_nodes = "localhost:8500"
es_index = "covid_data"
es_type = "covid_record"

# Configure Spark Elasticsearch-Hadoop connector
conf = {
    "es.nodes": es_nodes,
    "es.nodes.wan.only": "true",
    "es.index.auto.create": "true",
    "es.mapping.id": "id"
}

# Write DataFrame to Elasticsearch
df_selected.write.format("org.elasticsearch.spark.sql") \
    .options(**conf) \
    .mode("overwrite") \
    .save("{}/{}".format(es_index, es_type))
