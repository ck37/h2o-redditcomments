package org.mlab.spark.h2o

import hex.deeplearning.{DeepLearningModel, DeepLearning}
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import org.apache.spark.examples.h2o.DemoUtils._
import org.apache.spark.h2o._
import org.apache.spark.mllib.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.rdd.SchemaRDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext, mllib}
import water.app.{ModelMetricsSupport, SparkContextSupport}


object RedditModel extends SparkContextSupport with ModelMetricsSupport {

  def main(args: Array[String]) {
    // Create SparkContext to execute application on Spark cluster
    val sc = new SparkContext( configure("Reddit Comment Predictor") )

    // Initialize H2O & SQL context
    import h2oContext._
    implicit val h2oContext = H2OContext.getOrCreate(sc)
    import sqlContext.implicits._
    implicit val sqlContext = new SQLContext(sc)

    // Data load & tokenize
    val srdd: SchemaRDD = sqlContext.sql("SELECT *")
    val h2oFrame: H2OFrame = h2oContext.asH2OFrame(srdd)

    // TODO featurize data from table
    // ...

    // Split table into train-val
    val keys = Array[String]("train.hex", "valid.hex")
    val ratios = Array[Double](0.8)
    val frs = split(table, keys, ratios)
    val (train, valid) = (frs(0), frs(1))
    table.delete()

    // Build a model
    val dlModel = buildDLModel(train, valid)

    // Collect model metrics
    val trainMetrics = binomialMM(dlModel, train)
    val validMetrics = binomialMM(dlModel, valid)
    println(
      s"""
         |AUC on train data = ${trainMetrics.auc}
         |AUC on valid data = ${validMetrics.auc}
       """.stripMargin)

    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext=true)
  }


  /** Builds DeepLearning model. */
  def buildDLModel(train: Frame, valid: Frame,
                   epochs: Int = 10, l1: Double = 0.001, l2: Double = 0.0,
                   hidden: Array[Int] = Array[Int](200, 200))
                  (implicit h2oContext: H2OContext): DeepLearningModel = {
    import h2oContext._
    // Build a model
    val dlParams = new DeepLearningParameters()
    dlParams._train = train
    dlParams._valid = valid
    dlParams._response_column = 'target
    dlParams._epochs = epochs
    dlParams._l1 = l1
    dlParams._hidden = hidden

    // Create a job
    val dl = new DeepLearning(dlParams, water.Key.make("dlModel.hex"))
    val dlModel = dl.trainModel.get

    // Compute metrics on both datasets
    dlModel.score(train).delete()
    dlModel.score(valid).delete()

    dlModel
  }


  /** Text tokenizer
    *
    * Produce a bag of word representing given message.
    *
    * @param data RDD of text messages
    * @return RDD of bag of words
    */
  def tokenize(data: RDD[String]): RDD[Seq[String]] = {
    val ignoredWords = Seq("the", "a", "", "in", "on", "at", "as", "not", "for")
    val ignoredChars = Seq(',', ':', ';', '/', '<', '>', '"', '.', '(', ')', '?', '-', '\'','!','0', '1')

    val texts = data.map( r=> {
      var smsText = r.toLowerCase
      for( c <- ignoredChars) {
        smsText = smsText.replace(c, ' ')
      }

      val words =smsText.split(" ").filter(w => !ignoredWords.contains(w) && w.length>2).distinct

      words.toSeq
    })
    texts
  }

  /** Buil tf-idf model representing a text message. */
  def buildIDFModel(tokens: RDD[Seq[String]],
                    minDocFreq:Int = 4,
                    hashSpaceSize:Int = 1 << 10): (HashingTF, IDFModel, RDD[mllib.linalg.Vector]) = {
    // Hash strings into the given space
    val hashingTF = new HashingTF(hashSpaceSize)
    val tf = hashingTF.transform(tokens)
    // Build term frequency-inverse document frequency
    val idfModel = new IDF(minDocFreq = minDocFreq).fit(tf)
    val expandedText = idfModel.transform(tf)
    (hashingTF, idfModel, expandedText)
  }


}