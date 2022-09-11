//name of the package
name := "hohoapp"
//version of our package
version := "1.0"
//version of Scala
scalaVersion := "2.12.15"
// spark library dependencies 
// change this to 3.0.0 when released
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0",
  "org.apache.spark" %% "spark-sql"  % "3.3.0"
)
