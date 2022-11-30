from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
optinos = {
    "url": "jdbc:mysql://localhost:3306/sparktest?useSSL=false",
    'driver': 'com.mysql.cj.jdbc.Driver',
    'dbtable': 'people',
    'user': 'root',
    'password': '12345678'
}
df = spark.read.format('jdbc').options(**optinos).load()
print(f'df的类型: {type(df)}')
df.show()