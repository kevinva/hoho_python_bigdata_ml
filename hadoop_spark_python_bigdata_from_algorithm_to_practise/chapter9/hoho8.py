from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

data = [{'course': 'math', 'score': 80}, 
        {'course': 'math', 'score': 98},
        {'course': 'english', 'score': 85},
        {'course': 'english', 'score': 60}]

course_list = spark.createDataFrame(data).registerTempTable("course_list")
df = spark.sql("select course, sum(score) from course_list group by course")
hdfs_path = "hdfs://localhost:9000/bigdata/testdata/course_score1"
df.write.json(hdfs_path, mode = 'overwrite')