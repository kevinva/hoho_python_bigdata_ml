from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
data = [{'course': 'math', 'score': 80}, 
        {'course': 'math', 'score': 98},
        {'course': 'english', 'score': 85},
        {'course': 'english', 'score': 60}]
course_list = spark.createDataFrame(data).registerTempTable("course_list")
df = spark.sql("select course, sum(score) from course_list group by course")
properties = {
    'driver': 'com.mysql.cj.jdbc.Driver',
    'user': 'root',
    'password': '12345678'
}
df.write.jdbc('jdbc:mysql://localhost:3306/sparktest?useUnicode=true&characterEncoding=utf8&useSSL=false', table = 'course_list', mode = 'overwrite', properties = properties)