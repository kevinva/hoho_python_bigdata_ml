from pyspark.ml import pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()
train_data = spark.createDataFrame([
    ('I like spark ', 1.0),
    ('hbase hive', 0.0),
    ('spark good spark nice hello spark', 1.0),
    ('hbase hadoop hive', 0.0)
], ['text', 'label'])

test_data = spark.createDataFrame([
    ('spark not bad',),
    ('spark is ok',),
    ('hbase',),
    ('apache hadoop hive',)
], ['text'])

tokenizer = Tokenizer(inputCol = 'text', outputCol = 'words')
tokenizer_output = tokenizer.transform(train_data)
print('======Tokenizer output:')
tokenizer_output.show()

hashingTF = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol = 'features')
hashingTF_output = hashingTF.transform(tokenizer_output)
print('======HashingTF output:')
hashingTF_output.show()

lr = LogisticRegression(maxIter = 10, regParam = 0.001)
model = lr.fit(hashingTF_output)
print('======Model prediction:')
test_output = tokenizer.transform(test_data)
test_output = hashingTF.transform(test_output)
prediction = model.transform(test_output)
prediction.show()


print('======All in one:')
pipeline = pipeline.Pipeline(stages = [tokenizer, hashingTF, lr])
model_pipeline = pipeline.fit(train_data)
prediction_pipeline = model_pipeline.transform(test_data)
selected = prediction_pipeline.select('text', 'probability', 'prediction')
for row in selected.collect():
    text, prob, pred = row
    print(f'({text}) --> prob = {prob}, prediction = {pred}')