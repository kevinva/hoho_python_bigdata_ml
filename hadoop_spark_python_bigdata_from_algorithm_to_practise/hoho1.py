from pyspark import SparkContext

sc = SparkContext()
# rdd1 = sc.parallelize([1, 2, 3, 4, 5, 6])
# rdd2 = rdd1.map(lambda x: x * 2)
# local_data = rdd2.collect()
# [print(f'当前元素: {x}') for x in local_data]

# rdd1 = sc.parallelize(['lesson1 spark', 'lesson2 hadoop', 'lesson3 hive'])
# rdd2 = rdd1.flatMap(lambda x: x.split(' '))
# local_data = rdd2.collect()
# [print(f'当前元素: {x}') for x in local_data]

rdd1 = sc.parallelize([1, 2, 3, 4, 5, 6])
rdd2 = rdd1.filter(lambda x: x > 3)
local_data = rdd2.collect()
[print(f'当前元素是: {x}') for x in local_data]