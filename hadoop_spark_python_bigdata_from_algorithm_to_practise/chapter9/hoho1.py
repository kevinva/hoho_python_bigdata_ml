from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# df = spark.read.load('F:\\spark-3.3.0-bin-hadoop3\\examples\\src\\main\\resources\\users.parquet')
# df = spark.read.load('hdfs://localhost:9000/bigdata/testdata/users.parquet')
df = spark.read.load('F:\\spark-3.3.0-bin-hadoop3\\examples\\src\\main\\resources\\people.json', format='json')
print(f'df的类型: {type(df)}')
df.show()