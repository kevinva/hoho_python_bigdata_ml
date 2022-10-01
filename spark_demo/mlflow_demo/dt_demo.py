from multiprocessing.connection import Pipe
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import pandas as pd

spark = SparkSession.builder.appName('dt_mlflow_test').getOrCreate()

filePath = "/Users/kevinho/hohoho/temp/hoho_python_bigdata_ml/spark_demo/data/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)
categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == 'string']
indexOutputCols = [x + 'index' for x in categoricalCols]
stringIndexer = StringIndexer(inputCols = categoricalCols,
                              outputCols = indexOutputCols,
                              handleInvalid = 'skip')
numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == 'double') & (field != 'price'))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols = assemblerInputs,
                               outputCol = 'features')
rf = RandomForestRegressor(labelCol = 'price', maxBins = 40, maxDepth = 5, numTrees = 100, seed = 42)
pipeline = Pipeline(stages=[stringIndexer, vecAssembler, rf])

with mlflow.start_run(run_name='random-forest') as run:
    mlflow.log_param('num_trees', rf.getNumTrees())
    mlflow.log_param('max_depth', rf.getMaxDepth())

    pipelineModel = pipeline.fit(trainDF)
    mlflow.spark.log_model(pipelineModel, 'model')

    predDF = pipelineModel.transform(testDF)
    regressionEvaluator = RegressionEvaluator(predictionCol='prediction', labelCol='price')
    rmse = regressionEvaluator.setMetricName('rmse').evaluate(predDF)
    r2 = regressionEvaluator.setMetricName('r2').evaluate(predDF)
    mlflow.log_metrics({'rmse': rmse, 'r2': r2})
    print(f'rmse: {rmse}, r2: {r2}')

    rfModel = pipelineModel.stages[-1]
    pandasDF = pd.DataFrame(list(zip(vecAssembler.getInputCols(), rfModel.featureImportances)),
                            columns = ['feature', 'importance']).sort_values(by='importance', ascending = False)

    pandasDF.to_csv('feature-importance.csv', index = False)
    mlflow.log_artifact('feature-importance.csv')