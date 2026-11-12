ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "RoadAccidentPrediction",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql"   % "3.5.5",
      "org.apache.spark" %% "spark-mllib" % "3.5.5"
    ),
    // Spark on JDK 17 needs these flags when started from sbt
    run / fork := true,
    run / javaOptions ++= Seq(
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED"
    )
  )
