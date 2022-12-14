from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

data = [{'course': 'math', 'score': 80}, 
        {'course': 'math', 'score': 98},
        {'course': 'english', 'score': 85},
        {'course': 'english', 'score': 60}]

course_list = spark.createDataFrame(data).registerTempTable("course_list")
tmp_list = spark.sql("select course, max(score) from course_list group by course").collect()
[print(f'当前元素是：{item}') for item in tmp_list]