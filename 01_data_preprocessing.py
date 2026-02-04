"""
PySpark Data Preprocessing Script
Crop Yield Prediction and Recommendation System
"""

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, isnan, count, mean, stddev, min as spark_min, 
    max as spark_max, trim, upper, regexp_replace, monotonically_increasing_id,
    year, month, to_date, lit
)
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Crop Yield Data Preprocessing") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("=" * 80)
print("CROP YIELD PREDICTION - DATA PREPROCESSING")
print("=" * 80)

# ============================================================================
# 1. READING DATA
# ============================================================================
print("\n[1] READING DATA...")
data_path = "APY.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

print(f"✓ Data loaded successfully!")
print(f"  Total Records: {df.count():,}")
print(f"  Total Columns: {len(df.columns)}")

# Display schema
print("\n[SCHEMA]")
df.printSchema()

# Show sample data
print("\n[SAMPLE DATA]")
df.show(5, truncate=False)

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n[2] DATA EXPLORATION...")

# Check for missing values
print("\n[MISSING VALUES COUNT]")
missing_counts = df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) 
                             for c in df.columns])
missing_counts.show()

# Statistical summary
print("\n[STATISTICAL SUMMARY]")
df.describe().show()

# ============================================================================
# 3. DATA CLEANING
# ============================================================================
print("\n[3] DATA CLEANING...")

# 3.1 Clean column names (remove extra spaces)
print("\n  [3.1] Cleaning column names...")
df = df.toDF(*[c.strip() for c in df.columns])

# 3.2 Handle missing values
print("\n  [3.2] Handling missing values...")
initial_count = df.count()

# Remove rows where Yield is null (target variable)
df = df.filter(col("Yield").isNotNull())

# Remove rows where Production is null
df = df.filter(col("Production").isNotNull())

# Fill missing Area values with 0 or drop them
df = df.filter(col("Area").isNotNull())

# Clean State and District - remove nulls
df = df.filter(col("State").isNotNull())
df = df.filter(col("District").isNotNull())
df = df.filter(col("Crop").isNotNull())
df = df.filter(col("Season").isNotNull())

final_count = df.count()
print(f"  ✓ Removed {initial_count - final_count:,} rows with missing critical values")
print(f"  ✓ Remaining records: {final_count:,}")

# 3.3 Remove duplicates
print("\n  [3.3] Removing duplicates...")
initial_count = df.count()
df = df.dropDuplicates()
final_count = df.count()
print(f"  ✓ Removed {initial_count - final_count:,} duplicate rows")
print(f"  ✓ Remaining records: {final_count:,}")

# 3.4 Data type conversions
print("\n  [3.4] Converting data types...")
df = df.withColumn("Area", col("Area").cast(DoubleType()))
df = df.withColumn("Production", col("Production").cast(DoubleType()))
df = df.withColumn("Yield", col("Yield").cast(DoubleType()))
df = df.withColumn("Crop_Year", col("Crop_Year").cast(IntegerType()))

# Clean string columns
df = df.withColumn("State", trim(col("State")))
df = df.withColumn("District", trim(col("District")))
df = df.withColumn("Crop", trim(col("Crop")))
df = df.withColumn("Season", trim(col("Season")))

print("  ✓ Data types converted successfully")

# 3.5 Remove outliers (using IQR method for Yield)
print("\n  [3.5] Handling outliers...")

# Filter out negative or zero yields
df = df.filter(col("Yield") > 0)
df = df.filter(col("Area") > 0)
df = df.filter(col("Production") > 0)

# Remove extreme outliers (Yield > 1000 is unrealistic for most crops)
df = df.filter(col("Yield") < 1000)

print(f"  ✓ Outliers handled. Remaining records: {df.count():,}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n[4] FEATURE ENGINEERING...")

# 4.1 Create Season mapping to months
print("\n  [4.1] Creating season-to-month mapping...")
season_month_map = {
    "Kharif": 7,      # July (monsoon season)
    "Rabi": 11,       # November (winter season)
    "Summer": 3,      # March
    "Autumn": 9,      # September
    "Whole Year": 1   # January (default)
}

# Create month column based on season
from pyspark.sql.functions import create_map, lit
from itertools import chain

mapping_expr = create_map([lit(x) for x in chain(*season_month_map.items())])
df = df.withColumn("Month", mapping_expr[col("Season")])

print("  ✓ Month column created from Season")

# 4.2 Create productivity category
print("\n  [4.2] Creating productivity categories...")
df = df.withColumn("Productivity_Category",
    when(col("Yield") < 1, "Low")
    .when((col("Yield") >= 1) & (col("Yield") < 3), "Medium")
    .when((col("Yield") >= 3) & (col("Yield") < 10), "High")
    .otherwise("Very High")
)

print("  ✓ Productivity categories created")

# 4.3 Create decade column
print("\n  [4.3] Creating decade feature...")
df = df.withColumn("Decade", (col("Crop_Year") / 10).cast(IntegerType()) * 10)

print("  ✓ Decade feature created")

# 4.4 Add unique ID
df = df.withColumn("ID", monotonically_increasing_id())

print("\n[FEATURE ENGINEERING COMPLETE]")
df.printSchema()

# ============================================================================
# 5. DATA ENCODING (for ML)
# ============================================================================
print("\n[5] PREPARING DATA FOR MACHINE LEARNING...")

# Create indexed columns for categorical variables
print("\n  [5.1] Encoding categorical variables...")

# String Indexing
state_indexer = StringIndexer(inputCol="State", outputCol="State_Index")
district_indexer = StringIndexer(inputCol="District", outputCol="District_Index")
crop_indexer = StringIndexer(inputCol="Crop", outputCol="Crop_Index")
season_indexer = StringIndexer(inputCol="Season", outputCol="Season_Index")

# Fit and transform
df = state_indexer.fit(df).transform(df)
df = district_indexer.fit(df).transform(df)
df = crop_indexer.fit(df).transform(df)
df = season_indexer.fit(df).transform(df)

print("  ✓ Categorical encoding complete")

# ============================================================================
# 6. SAVE CLEANED DATA
# ============================================================================
print("\n[6] SAVING CLEANED DATA...")

# Save as Parquet (efficient for big data)
output_path = "outputs/cleaned_data.parquet"
df.write.mode("overwrite").parquet(output_path)
print(f"  ✓ Data saved to: {output_path}")

# Also save as CSV for Streamlit
csv_output_path = "outputs/cleaned_data.csv"
df.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_output_path)
print(f"  ✓ Data saved to: {csv_output_path}")

# ============================================================================
# 7. DATA SUMMARY STATISTICS
# ============================================================================
print("\n[7] FINAL DATA SUMMARY...")

print("\n[RECORD COUNT BY CROP (Top 20)]")
df.groupBy("Crop").count().orderBy(col("count").desc()).show(20)

print("\n[RECORD COUNT BY STATE (Top 15)]")
df.groupBy("State").count().orderBy(col("count").desc()).show(15)

print("\n[RECORD COUNT BY SEASON]")
df.groupBy("Season").count().orderBy(col("count").desc()).show()

print("\n[AVERAGE YIELD BY CROP (Top 20)]")
df.groupBy("Crop").agg({"Yield": "mean"}).withColumnRenamed("avg(Yield)", "Avg_Yield") \
    .orderBy(col("Avg_Yield").desc()).show(20)

print("\n[AVERAGE YIELD BY STATE (Top 15)]")
df.groupBy("State").agg({"Yield": "mean"}).withColumnRenamed("avg(Yield)", "Avg_Yield") \
    .orderBy(col("Avg_Yield").desc()).show(15)

print("\n[YIELD STATISTICS BY SEASON]")
df.groupBy("Season").agg(
    mean("Yield").alias("Mean_Yield"),
    stddev("Yield").alias("StdDev_Yield"),
    spark_min("Yield").alias("Min_Yield"),
    spark_max("Yield").alias("Max_Yield")
).show()

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFinal Dataset Statistics:")
print(f"  Total Records: {df.count():,}")
print(f"  Total Features: {len(df.columns)}")
print(f"  Unique Crops: {df.select('Crop').distinct().count()}")
print(f"  Unique States: {df.select('State').distinct().count()}")
print(f"  Unique Districts: {df.select('District').distinct().count()}")
print(f"  Year Range: {df.agg(spark_min('Crop_Year')).collect()[0][0]} - {df.agg(spark_max('Crop_Year')).collect()[0][0]}")
print("\n" + "=" * 80)

# Stop Spark session
spark.stop()
