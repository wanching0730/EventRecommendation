import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id

object MovieRecommendation {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[*]").appName("Spark Word Count").getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._ // convert RDD to DF

    // Convert RDD to DF
    val ratings = sc.textFile("s3://aws-emr-resources-096889944571-us-east-2/file.txt")

    val ratingSplit = ratings.map(_.split("\n"))

    val ratingFlatten = ratingSplit.flatMap(x => x)

    val ratingDF = ratingFlatten.map(arr => {
        val newArr = arr.split(",")
        Rating(newArr(0).toInt, newArr(1).toInt, newArr(2).toFloat)
    }).toDF()


    // Build the recommendation model using ALS on the training data
    val Array(training, test) = ratingDF.randomSplit(Array(0.8, 0.2))
    val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("eventId").setRatingCol("rating")
    val model = als.fit(training)

    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")


    // Generate top 10 event recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    userRecs.select("recommendations").show(false)


    // Cast all columns to String
    val df1 = userRecs.withColumn("userId", userRecs.col("userId").cast("string")).select("userId")
    val df2 = userRecs.withColumn("eventId", userRecs.col("recommendations.eventId").cast("string")).select("eventId")
    val df3 = userRecs.withColumn("rating", userRecs.col("recommendations.rating").cast("string")).select("rating")

    val df4 = df1.withColumn("id", monotonically_increasing_id()).join(df2.withColumn("id", monotonically_increasing_id()), Seq("id")).join(df3.withColumn("id", monotonically_increasing_id()), Seq("id")).drop("id")
    df4.show(false)

    // Output to S3 (use s3n)
    val outputFileUri = s"s3n://aws-emr-resources-096889944571-us-east-2/output.csv/"
    df4.repartition(1).write.format("csv").mode("overwrite").option("header", "true").save(outputFileUri)
    //df4.repartition(1).write.csv(outputFileUri)

    val ratings1 = sc.textFile("s3://useis-prediction/file1.txt")

    val ratingSplit1 = ratings1.map(_.split("\n"))

    val ratingFlatten1 = ratingSplit1.flatMap(x => x)

    val ratingDF1 = ratingFlatten1.map(arr => {
        val newArr = arr.split(",")
        Rating(newArr(0).toInt, newArr(1).toInt, newArr(2).toFloat)
    }).toDF()


    val Array(training, test) = ratingDF1.randomSplit(Array(0.8, 0.2))
    val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("eventId").setRatingCol("rating")
    val model = als.fit(training)


    val predictions1 = model.transform(test)

    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
    val rmse1 = evaluator.evaluate(predictions1)
    println(s"Root-mean-square error = $rmse1")

    // Generate top 10 event recommendations for each user
    val userRecs1 = model.recommendForAllUsers(10)
    userRecs1.select("recommendations").show(false)

    // Cast all columns to String
    val df4 = userRecs1.withColumn("userId", userRecs1.col("userId").cast("string")).select("userId")
    val df5 = userRecs1.withColumn("eventId", userRecs1.col("recommendations.eventId").cast("string")).select("eventId")
    val df6 = userRecs1.withColumn("rating", userRecs1.col("recommendations.rating").cast("string")).select("rating")

    val df7 = df4.withColumn("id", monotonically_increasing_id()).join(df5.withColumn("id", monotonically_increasing_id()), Seq("id")).join(df6.withColumn("id", monotonically_increasing_id()), Seq("id")).drop("id")
    df7.show(false)

    // Output to S3 (use s3n)
    val outputFileUri = s"s3n://useis-prediction/output1.csv"
    df7.repartition(1).write.format("csv").mode("overwrite").option("header", "true").save(outputFileUri)
    //df4.repartition(1).write.csv(outputFileUri)
}
