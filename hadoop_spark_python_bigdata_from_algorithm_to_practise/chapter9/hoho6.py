from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
data = [{'name': 'Alice', 'age': 1}, {'name': 'Bob', 'age': 2}, {'name': 'Li', 'age': 3}]
df = spark.createDataFrame(data)
spark.udf.register("show_name", lambda item: "姓名是：" + item)
tmp_list = df.selectExpr("show_name(name)", "age + 1").collect()
[print("当前元素是:", item) for item in tmp_list]
