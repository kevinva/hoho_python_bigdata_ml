from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
data = [{'name': 'Alice', 'age': 1}, {'name': 'Bob', 'age': 2}, {'name': 'Li', 'age': 3}]
df = spark.createDataFrame(data)

# # df.show()
# tmp_list = df.filter("'name' = 'Alice' and 'age' = 1").collect()
# # [print(f'当前元素是：｛item｝') for item in tmp_list]
# print(tmp_list)

tmp_list = df.select('name').collect()
[print('当前元素是:', item) for item in tmp_list]