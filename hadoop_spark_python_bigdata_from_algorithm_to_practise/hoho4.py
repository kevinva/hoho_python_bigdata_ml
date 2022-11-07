from pyspark import SparkContext

sc = SparkContext()
rdd1 = sc.textFile("/bigdata/a_seafood.txt")

# map
def func(item):
    data = item.split(":")
    return data[0], data[1]

rdd2 = rdd1.map(func)
result = rdd2.collect()

def f(item):
    print(f"当前元素是: {item}")

[f(item) for item in result]

# flatMap
rdd1 = sc.parallelize(["黑虎虾, 扇贝, 黄花鱼, 鲈鱼, 罗非鱼, 鲜贝, 阿根廷红虾"])
rdd2 = rdd1.flatMap(lambda item: item.split(","))
result = rdd2.collect()
[f(item) for item in result]


# lookup
rdd1 = sc.textFile("/bigdata/a_seafood.txt")
rdd2 = rdd1.map(func)
result = rdd2.lookup("黑虎虾")
print(f"当前元素是: {result}")

# zip
rdd1 = sc.parallelize([139, 16.9, 49.9, 35.9, 29.9], 3)  # 最后的3表示分区数
rdd2 = sc.parallelize(["黑虎虾", "扇贝", "黄花鱼", "鲈鱼", "罗非鱼"], 3)
result = rdd2.zip(rdd1).collect()
[f(item) for item in result]

# join
rdd1 = sc.parallelize([("黑虎虾", 100), ("扇贝", 10.2), ("鲈鱼", 55.9)])
rdd2 = sc.parallelize([("黑虎虾", 139), ("扇贝", 16.9), ("黄花鱼", 35.9), ("罗非鱼", 29.9)])
result = rdd1.join(rdd2).collect()
print(f'join的结果是: {result}')

# leftOuterJoin
result = rdd1.leftOuterJoin(rdd2).collect()
[f(item) for item in result]

# fullOuterJoin
result = rdd1.fullOuterJoin(rdd2).collect()
[f(item) for item in result]

# combineByKey
rdd = sc.parallelize([("黑虎虾", 139), ("黑虎虾", 100), ("扇贝", 16.9), ("扇贝", 10.2), ("海参", 59.9), ("鲈鱼", 35.9), ("罗非鱼", 29.9)])

def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a
    
result = rdd.combineByKey(to_list, append, extend).collect()
[f(item) for item in result]
