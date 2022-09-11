package src.main.scala

import org.apache.spark.{SparkConf, SparkContext}

object AvgAgeCalculator {
  def main(args: Array[String]) {

    //先本地测试
    val sparkConf = new SparkConf().setAppName("AvgAgeCalculator").setMaster("local[2]")

    //再调整到集群上测试
    // val sparkConf = new SparkConf().setAppName("AvgAgeCalculator")

    val sc = new SparkContext(sparkConf)

    val dataFile = sc.textFile("file:///Users/kevinho/hohoho/temp/hoho_python_bigdata_ml/spark_demo/sparkapp/data/sample_age_data.txt")

    val ageData = dataFile.map(line => line.split(" ")(1)) //第一步
    val count = dataFile.count() //第二步
    val totalAge = ageData.map(age => age.toInt).reduce(_ + _) //第三步
    val avgAge = totalAge / count //第四步

    println("TotalAge is: " + totalAge +
      " , Number of People: " + count +
      " , Average Age is : " + avgAge)

    sc.stop()
  }
}