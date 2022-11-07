from pyspark import SparkContext

sc = SparkContext()
a_rdd = sc.textFile('/bigdata/a_seal.txt')
b_rdd = sc.textFile('/bigdata/b_seal.txt')
union_rdd = a_rdd.union(b_rdd) 

def f(item):
    tmp = item.split(':')
    return tmp[0], int(tmp[1])

map_rdd = union_rdd.map(f)
map_rdd.cache()

def create_combiner(v):
    return v, 1

def merge_value(c, v):
    return c[0] + v, c[1] + 1

def merge_combiner(c1, c2):
    return c1[0] + c2[0], c1[1] + c2[1]

rdd = map_rdd.combineByKey(create_combiner, merge_value, merge_combiner)
result = map_rdd.reduceByKey(lambda a, b: a + b).collect() # 求总销量 # 如果之前不缓存map_rdd，这里会重复计算map_rdd
print('总销量：')
[print("当前元素是: ", item) for item in result]

print('=============')
print('平均销量：')
result = rdd.map(lambda x: (x[0], x[1][0] / x[1][1])).collect()  # 求平均
[print("当前元素是: ", item) for item in result]


print('=============')
print('销量排名：')
a_map_rdd = a_rdd.map(f)
b_map_rdd = b_rdd.map(f)
join_rdd = a_map_rdd.join(b_map_rdd)

def f2(item):
    return item[0], sum(item[1])
result = join_rdd.map(f2).sortBy(lambda x: x[1], False).collect()
[print(item) for item in result]