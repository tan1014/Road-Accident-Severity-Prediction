# Road Accident Severity Prediction (Scala + Apache Spark)

Classroom project for INT315 Cluster Computing. A Spark MLlib pipeline that predicts
road accident severity (Slight, Serious or Fatal Injury) from driver, vehicle, road and
environment attributes, using a Decision Tree classifier.

## Status

The code is written but has not yet been run end to end on the Addis Ababa dataset.
No accuracy figures are claimed. Results will be added to this README after the first run.

## What the program does

1. Loads `data/RTA Dataset.csv` into a Spark DataFrame
2. Cleans it (trims text, turns blank and NA values into null, drops rows with no severity)
3. Explores it with Spark SQL (severity, weather and road-surface counts)
4. Indexes the 10 categorical features and assembles them into a feature vector
5. Trains a Decision Tree (`maxDepth = 5`) on an 80/20 split (seed 42)
6. Prints actual vs predicted severity, accuracy, F1, weighted precision and recall, and a confusion matrix
7. Saves the trained pipeline to `models/road_accident_model`

## Requirements

- JDK 17
- Scala 2.12.18 and sbt (versions are set in `build.sbt` and `project/build.properties`)
- Apache Spark 3.5.5 (downloaded by sbt, no separate install)

## Run

1. Put `RTA Dataset.csv` in the `data/` folder (see `data/README.md`).
2. From the project folder run `sbt run`, or open the project in IntelliJ IDEA and run `Main`.

If you run from IntelliJ and see an `IllegalAccessError` about `sun.nio.ch`, add the
`--add-opens` options from `build.sbt` to the run configuration's VM options.

## Project layout

```
RoadAccidentPrediction/
├── build.sbt
├── project/build.properties
├── data/                 (put RTA Dataset.csv here)
└── src/main/scala/Main.scala
```

## Notes and limitations

- Runs in Spark local mode (`local[*]`) on one machine.
- Column names in `Main.scala` must match the CSV header exactly (for example `Road_allignment`).
- Educational prototype, not a validated road-safety system.
