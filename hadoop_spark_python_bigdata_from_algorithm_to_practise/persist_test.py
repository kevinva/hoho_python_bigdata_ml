from pyspark import SparkContext, StorageLevel

sc = SparkContext()

data = [1, 2, 3, 4, 5, 6]

def show(item):
    print(f'当前元素是: {item}')
    return item * 2

rdd = sc.parallelize(data, 4).map(lambda x: show(x))
rdd.persist(StorageLevel.MEMORY_ONLY)
print(f'获取最小值: {rdd.min()}')
print(f'获取最大值: {rdd.max()}')