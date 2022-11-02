from pyspark import SparkContext
from operator import add

sc = SparkContext()
rdd = sc.parallelize(['Spark', 'hadoop', 'hive'])
result = rdd.count()
print(f'1. count: 当前RDD的元素个数是: {result}')

rdd = sc.parallelize([('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)])
result = rdd.sortBy(lambda x: x[1], False).first()
print(f'2. first/sortBy: 当前元素是: {result}')

result = rdd.sortBy(lambda x: x[1], False).take(3)
print(f'3. take/sortBy: 当前元素是: {result}')

result = rdd.map(lambda x: x[1]).reduce(add)
print(f'4. reduce/map: 当前元素是: {result}')

rdd = sc.parallelize([('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)], 2)

def f(x):
    print(f'5. foreach: 当前数据项: {x}')
result = rdd.foreach(f)

def f1(iterator):
    print(list(iterator))
result = rdd.foreachPartition(f1)