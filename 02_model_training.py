"""
PySpark Machine Learning Model Training
Crop Yield Prediction using Random Forest Regressor
"""

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import json

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Crop Yield ML Model") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("=" * 80)
print("CROP YIELD PREDICTION - MACHINE LEARNING MODEL TRAINING")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] LOADING PREPROCESSED DATA...")
df = spark.read.parquet("outputs/cleaned_data.parquet")

print(f"✓ Data loaded successfully!")
print(f"  Total Records: {df.count():,}")

# ============================================================================
# 2. FEATURE SELECTION
# ============================================================================
print("\n[2] FEATURE SELECTION...")

# Select features for the model
# We'll use: State, District, Crop, Season, Area, Crop_Year, Month
feature_cols = ["State_Index", "District_Index", "Crop_Index", "Season_Index", 
                "Area", "Crop_Year", "Month"]

# Target variable
target_col = "Yield"

# Filter data to ensure all required columns exist
required_cols = feature_cols + [target_col]
df_ml = df.select(*required_cols).na.drop()

print(f"  ✓ Selected {len(feature_cols)} features")
print(f"  ✓ Target variable: {target_col}")
print(f"  ✓ Records after removing nulls: {df_ml.count():,}")

# ============================================================================
# 3. PREPARE FEATURES
# ============================================================================
print("\n[3] PREPARING FEATURES...")

# Assemble features into a single vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_ml = assembler.transform(df_ml)
print("  ✓ Features assembled into vector")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4] SPLITTING DATA...")

# Split data: 80% training, 20% testing
train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)

print(f"  ✓ Training set: {train_data.count():,} records")
print(f"  ✓ Test set: {test_data.count():,} records")

# ============================================================================
# 5. MODEL TRAINING - RANDOM FOREST
# ============================================================================
print("\n[5] TRAINING RANDOM FOREST MODEL...")

# Initialize Random Forest Regressor
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=target_col,
    numTrees=100,
    maxDepth=10,
    seed=42
)

# Train the model
print("  Training in progress...")
rf_model = rf.fit(train_data)
print("  ✓ Random Forest model trained successfully!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n[6] EVALUATING MODEL...")

# Make predictions on test data
predictions = rf_model.transform(test_data)

# Initialize evaluators
rmse_evaluator = RegressionEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="r2"
)

# Calculate metrics
rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print("\n[MODEL PERFORMANCE METRICS]")
print("=" * 50)
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Mean Absolute Error (MAE):      {mae:.4f}")
print(f"  R² Score:                       {r2:.4f}")
print("=" * 50)

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n[7] FEATURE IMPORTANCE...")

# Get feature importances
feature_importance = rf_model.featureImportances.toArray()

# Create feature importance dictionary
importance_dict = {}
for idx, importance in enumerate(feature_importance):
    importance_dict[feature_cols[idx]] = float(importance)

# Sort by importance
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\n[FEATURE IMPORTANCE RANKING]")
print("=" * 50)
for feature, importance in sorted_importance:
    print(f"  {feature:20s}: {importance:.4f}")
print("=" * 50)

# ============================================================================
# 8. SAMPLE PREDICTIONS
# ============================================================================
print("\n[8] SAMPLE PREDICTIONS...")

# Show sample predictions
print("\n[ACTUAL vs PREDICTED (Sample)]")
predictions.select(target_col, "prediction").show(20)

# ============================================================================
# 9. SAVE MODEL AND METRICS
# ============================================================================
print("\n[9] SAVING MODEL AND METRICS...")

# Save the model
model_path = "models/yield_prediction_model"
rf_model.write().overwrite().save(model_path)
print(f"  ✓ Model saved to: {model_path}")

# Save metrics to JSON
metrics = {
    "model_type": "Random Forest Regressor",
    "num_trees": 100,
    "max_depth": 10,
    "rmse": rmse,
    "mae": mae,
    "r2_score": r2,
    "training_records": train_data.count(),
    "test_records": test_data.count(),
    "features": feature_cols,
    "feature_importance": importance_dict
}

metrics_path = "models/model_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"  ✓ Metrics saved to: {metrics_path}")

# ============================================================================
# 10. ADDITIONAL ANALYSIS
# ============================================================================
print("\n[10] ADDITIONAL ANALYSIS...")

# Calculate prediction errors
predictions_with_error = predictions.withColumn(
    "error",
    col("prediction") - col(target_col)
).withColumn(
    "absolute_error",
    abs(col("prediction") - col(target_col))
)

print("\n[ERROR STATISTICS]")
predictions_with_error.select("error", "absolute_error").describe().show()

# ============================================================================
# 11. CROP-WISE PERFORMANCE
# ============================================================================
print("\n[11] ANALYZING CROP-WISE PERFORMANCE...")

# Join with original data to get crop names
predictions_with_crops = predictions.join(
    df.select("State_Index", "Crop_Index", "Crop"),
    on=["State_Index", "Crop_Index"],
    how="left"
)

print("\n[AVERAGE PREDICTION ERROR BY CROP (Top 20)]")
predictions_with_crops.groupBy("Crop").agg(
    {"error": "mean"}
).withColumnRenamed("avg(error)", "Avg_Error").orderBy(
    abs(col("Avg_Error")).desc()
).show(20)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"  Algorithm: Random Forest Regressor")
print(f"  Number of Trees: 100")
print(f"  Max Depth: 10")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R² Score: {r2:.4f}")
print(f"\n  Model saved at: {model_path}")
print(f"  Metrics saved at: {metrics_path}")
print("\n" + "=" * 80)

# Stop Spark session
spark.stop()
