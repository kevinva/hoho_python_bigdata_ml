from pyspark.sql import SparkSession

# spark = SparkSession.builder.getOrCreate()

# data = [{'name': 'Alice', 'age': 1}]
# df = spark.createDataFrame(data)
# print(df.printSchema)


spark = SparkSession.builder.getOrCreate()

data = [{'name': 'AliceAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'age': 1}, {'name': 'Bob', 'age': 3}]
df = spark.createDataFrame(data)
print(df.show(vertical=True))