from pyspark import SparkContext

sc = SparkContext()

# rdd1 = sc.parallelize([('a', 1), ('a', 1), ('a', 1), ('b', 1), ('b', 1), ('c', 1)])
# list1 = rdd1.groupByKey().mapValues(len).collect()
# [print(f'按key分组后的数据项: {item}') for item in list1]
# list2 = rdd1.groupByKey().mapValues(list).collect()
# [print(f'每一个key对应的数据: {item}') for item in list2]

# rdd1 = sc.parallelize(['Spark', 'Spark', 'hadoop', 'hadoop', 'hadoop', 'hadoop', 'hive'])
# rdd2 = rdd1.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).collect()
# [print(f'当前元素是: {item}') for item in rdd2]

# rdd1 = sc.parallelize(['Spark', 'hadoop', 'hive'])
# rdd2 = sc.parallelize(['Spark', 'kafka', 'hbase'])
# rdd3 = rdd1.union(rdd2).collect()
# print(f'合并结果: {rdd3}')

rdd1 = sc.parallelize(['Spark', 'hadoop', 'hive'])
rdd2 = sc.parallelize(['Spark', 'kafka', 'hbase'])
rdd3 = rdd1.union(rdd2).distinct().collect()
print(f'合并结果: {rdd3}')