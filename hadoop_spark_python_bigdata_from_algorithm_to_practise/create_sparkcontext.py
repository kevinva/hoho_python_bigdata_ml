from pyspark import SparkContext

sc = SparkContext()
# print(f'Spark Version: {sc.version}')

rdd = sc.textFile("file:///usr/local/spark/README.md")