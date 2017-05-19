import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val rdd0 = sc.textFile("/home/rodolfo/sense/bradesco/bradesco-20170510.txt")

val header = rdd0.first

case class Sense(id_sensor: String, date_time: String, id_campaign: String, vendor: String, mac_adress: String, date_time_timestamp: String, distance: String, rssi: Int, ssid: String)

val df1 = rdd0.filter(_!=header).map(s => s.split("\\|")).map(s => Sense(s(0).trim, s(1).split("\\.")(0) ,s(2).trim ,s(3).trim ,s(4).trim ,s(5).trim ,s(6).trim ,s(7).trim.toInt + 108 ,s(8).trim)).toDF.withColumn("date_time", unix_timestamp($"date_time").cast("timestamp"))


val df2 = df1.withColumn("pos",
                                when($"date_time" >= "2017-05-10 19:36:00" && $"date_time" <= "2017-05-10 19:45:00" ,"dentro")
                                .otherwise(
                                when($"date_time" >= "2017-05-10 19:46:00" && $"date_time" <= "2017-05-10 19:55:00" ,"fora")
                                .otherwise("semLabel")
                                )
                        )

val mac = Seq("84:10:0D:C5:30:AB", "7C:50:49:2C:51:E3", "1C:B7:2C:F1:51:48")
val df3 = df2.filter($"mac_adress".isin(mac: _*))
val df4 = df3.withColumn("ssid", when($"ssid" === "","Nenhuma").otherwise("uma"))

def remove: (String => String) = {s => s.dropRight(3)}
val removeDecimal = udf(remove)

def toStringFunc: (Timestamp => String) = {s => s.toString}
val to_string= udf(toStringFunc)

def toStringFunc2: (Int => String) = {s => s.toString}
val to_string2= udf(toStringFunc2)


val df5 = df4.withColumn("date_time", removeDecimal(to_string($"date_time")))

val df6 = df5.groupBy("date_time","mac_adress","vendor","id_Campaign","ssid","pos")
             .pivot("id_sensor")
             .agg(avg("rssi")).na.fill(0.0)
             

case class Posicao(macs: Int, x: Double, y: Double)

val mac_per_place = sc.parallelize(Seq(
    Posicao(3, 218.90, 799.41),
    Posicao(2, 213.79, 704.28),
    Posicao(1, 481.28, 682.80),
    Posicao(4, 820.38, 427.58)
    )).toDF

val maxIndex = (i: Double, j:Double, k:Double, l:Double) => {
    val all0 = List(i,j,k,l)
    val all1 = all0.sortWith(_ > _)
    all0.indexOf(all1(0)) + 1.0
}


val secondMaxIndex = (i: Double, j:Double, k:Double, l:Double) => {
    val all0 = List(i,j,k,l)
    val all1 = all0.sortWith(_ > _)
    
    if(all1(1) == 0.0) {
        all0.indexOf(all1(0)) + 1.0
    } else {
     all0.indexOf(all1(1)) + 1.0   
    }
}

val maxValueIndex = udf(maxIndex)
val secondMaxValueIndex = udf(secondMaxIndex)


val df7 = df6.withColumn("totem_entrada" , when($"18:D6:C7:43:29:7C" =!= 0, 1.0).otherwise(0.0))
             .withColumn("recepcao" , when($"60:E3:27:93:33:30" =!= 0, 1.0).otherwise(0.0))
             .withColumn("mesa_entrada" , when($"98:DE:D0:DB:BE:F8" =!= 0, 1.0).otherwise(0.0))
             .withColumn("fundos" , when($"98:DE:D0:DB:D3:BE" =!= 0, 1.0).otherwise(0.0))
             .withColumn("all_fundo", when($"fundos" === 1.0, "TRUE").otherwise("FALSE"))
             .withColumn("total_fundo", when($"fundos" === 1.0, 0.0).otherwise(1.0))
             .withColumn("all_entrada", when($"totem_entrada" =!= 0.0 && $"recepcao" =!= 0.0 && $"mesa_entrada"  =!= 0.0, "TRUE").otherwise("FALSE"))
             .withColumn("total_entrada", $"totem_entrada" + $"recepcao" + $"mesa_entrada")
             .withColumn("closest_sensor", maxValueIndex($"18:D6:C7:43:29:7C", $"60:E3:27:93:33:30", $"98:DE:D0:DB:BE:F8", $"98:DE:D0:DB:D3:BE"))
             .withColumn("second_closest_sensor", secondMaxValueIndex($"18:D6:C7:43:29:7C", $"60:E3:27:93:33:30", $"98:DE:D0:DB:BE:F8", $"98:DE:D0:DB:D3:BE"))
             

val df8 = df7.join(mac_per_place).where($"closest_sensor" === $"macs")
                                 .withColumn("x_closest", $"x")
                                 .withColumn("y_closest", $"y")
                                 .drop("macs","x","y")
                                 
val df9 = df8.join(mac_per_place).where($"second_closest_sensor" === $"macs")
                                 .withColumn("x_second_closest", $"x")
                                 .withColumn("y_second_closest", $"y")
                                 .drop("macs","x","y")                                 
                                 
val df10 = df9.withColumn("indexedLabel", when($"pos" === "dentro" , 0.0).otherwise(1.0))
              .withColumn("indexed_ssid", when($"ssid" === "Nenhuma" , 0.0).otherwise(1.0))
              .withColumn("indexed_allFundo", when($"all_fundo" === "FALSE" , 0.0).otherwise(1.0))
              .withColumn("indexed_allEntrada", when($"all_entrada" === "FALSE" , 0.0).otherwise(1.0))
              .drop("date_time","mac_adress","vendor","id_campaign", "ssid", "totem_entrada", "recepcao", "recepcao","mesa_entrada", "fundos", "all_fundo", "all_entrada")


import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler

val feat = df10.columns.filter(!_.contains("pos")).filter(!_.contains("Label"))

val assembler = new VectorAssembler().setInputCols(feat).setOutputCol("indexedFeatures")
val df11 = assembler.transform(df10)

val Array(trainingData, testData) = df11.randomSplit(Array(0.75, 0.25))


import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

val rf1 = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(60)
val rf2 = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(80)
val rf3 = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(100)

val model_rf1 = rf1.fit(trainingData)
val model_rf2 = rf2.fit(trainingData)
val model_rf3 = rf3.fit(trainingData)

val pred_rf1 = model_rf1.transform(testData)
val pred_rf2 = model_rf2.transform(testData)
val pred_rf3 = model_rf3.transform(testData)


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy1 = evaluator.evaluate(pred_rf1)
val accuracy2 = evaluator.evaluate(pred_rf2)
val accuracy3 = evaluator.evaluate(pred_rf3)


import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}

val gbt1 = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(3)
val gbt2 = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(6)
val gbt3 = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(9)

val model_gbt1 = gbt1.fit(trainingData)
val model_gbt2 = gbt2.fit(trainingData)
val model_gbt3 = gbt3.fit(trainingData)

val pred_gbt1 = model_gbt1.transform(testData)
val pred_gbt2 = model_gbt2.transform(testData)
val pred_gbt3 = model_gbt3.transform(testData)


val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy_gbt1 = evaluator.evaluate(pred_gbt1)
val accuracy_gbt2 = evaluator.evaluate(pred_gbt2)
val accuracy_gbt3 = evaluator.evaluate(pred_gbt3)







