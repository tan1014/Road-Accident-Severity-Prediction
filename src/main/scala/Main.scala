import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer,
  StringIndexerModel, IndexToString, VectorAssembler}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Main {
  def main(args: Array[String]): Unit = {

    // Part A: Spark session
    val spark = SparkSession.builder()
      .appName("Road Accident Severity Prediction")
      .master("local[*]")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    println("Road Accident Severity Prediction Started")

    // Part B: load the dataset
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "false")
      .option("mode", "DROPMALFORMED")
      .csv("data/RTA Dataset.csv")

    println("Total Records: " + df.count())

    println("Dataset Schema:")
    df.printSchema()

    println("First 10 Records:")
    df.show(10, false)

    // Part C: clean the dataset
    val cleaned = df.columns.foldLeft(df) {
      (data, columnName) =>
        data.withColumn(
          columnName,
          when(
            trim(col(columnName).cast("string")) === "" ||
            upper(trim(col(columnName).cast("string"))) === "NA",
            lit(null)
          ).otherwise(trim(col(columnName).cast("string")))
        )
    }

    val validData = cleaned.filter(
      col("Accident_severity").isNotNull
    )

    println("After Cleaning:")
    println("Records: " + validData.count())

    // Part D: exploration with Spark SQL
    validData.createOrReplaceTempView("accidents")

    println("Accident Severity Distribution:")

    spark.sql("""
      SELECT Accident_severity,
             COUNT(*) AS total
      FROM accidents
      GROUP BY Accident_severity
      ORDER BY total DESC
    """).show()

    println("Weather Conditions:")

    spark.sql("""
      SELECT Weather_conditions,
             COUNT(*) AS total
      FROM accidents
      GROUP BY Weather_conditions
      ORDER BY total DESC
    """).show()

    println("Road Surface Conditions:")

    spark.sql("""
      SELECT Road_surface_conditions,
             COUNT(*) AS total
      FROM accidents
      GROUP BY Road_surface_conditions
      ORDER BY total DESC
    """).show()

    // Part E: select features
    val featureColumns = Array(
      "Age_band_of_driver",
      "Sex_of_driver",
      "Driving_experience",
      "Type_of_vehicle",
      "Road_surface_conditions",
      "Weather_conditions",
      "Light_conditions",
      "Road_allignment",
      "Types_of_Junction",
      "Number_of_vehicles_involved"
    )

    val requiredColumns =
      featureColumns :+ "Accident_severity"

    val modelData = validData
      .select(requiredColumns.map(col): _*)
      .na.drop()

    println("Records available for Machine Learning:")
    println(modelData.count())

    // Part F: convert text to numbers
    val indexers = featureColumns.map { columnName =>
      new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(columnName + "_index")
        .setHandleInvalid("keep")
    }

    val labelIndexer = new StringIndexer()
      .setInputCol("Accident_severity")
      .setOutputCol("label")
      .setHandleInvalid("skip")

    val assembler = new VectorAssembler()
      .setInputCols(
        featureColumns.map(_ + "_index")
      )
      .setOutputCol("features")

    // Part G: pipeline, split and train
    val decisionTree = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(5)
      .setSeed(42)

    val pipeline = new Pipeline()
      .setStages(
        indexers ++
        Array(labelIndexer, assembler, decisionTree)
      )

    val Array(trainingData, testingData) =
      modelData.randomSplit(Array(0.8, 0.2), 42)

    println("Training Records: " + trainingData.count())
    println("Testing Records: " + testingData.count())

    val model = pipeline.fit(trainingData)
    println("Decision Tree Model Trained Successfully!")

    // Part H: predict severity
    val rawPredictions = model.transform(testingData)

    val labelModel =
      model.stages(featureColumns.length)
        .asInstanceOf[StringIndexerModel]

    println("Label order: " + labelModel.labels.mkString(", "))

    val converter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("Predicted_Severity")
      .setLabels(labelModel.labels)

    val predictions =
      converter.transform(rawPredictions)

    println("Actual vs Predicted Severity:")

    predictions.select(
      "Accident_severity",
      "Predicted_Severity"
    ).show(20, false)

    // Part I: evaluate the model
    val metrics = Array(
      ("Accuracy", "accuracy"),
      ("F1 Score", "f1"),
      ("Weighted Precision", "weightedPrecision"),
      ("Weighted Recall", "weightedRecall")
    )

    metrics.foreach { case (name, metric) =>
      val evaluator =
        new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName(metric)

      println(
        name + ": " + evaluator.evaluate(predictions)
      )
    }

    println("Confusion Matrix:")

    predictions
      .groupBy("label", "prediction")
      .count()
      .orderBy("label", "prediction")
      .show()

    // Part J: save the model and stop Spark
    model.write
      .overwrite()
      .save("models/road_accident_model")

    println("Model Saved Successfully!")

    spark.stop()
  }
}
