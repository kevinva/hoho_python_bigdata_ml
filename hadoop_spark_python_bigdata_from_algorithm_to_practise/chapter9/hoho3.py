from pyspark.sql import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark.sql('use sparktest')
df = spark.sql('select * from people')
print(f'df的类型: {type(df)}')
df.show()