"""
Simplified Data Preprocessing Script (Pandas-based)
Crop Yield Prediction and Recommendation System
This version works without PySpark/Java
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("=" * 80)
print("CROP YIELD PREDICTION - DATA PREPROCESSING (PANDAS VERSION)")
print("=" * 80)

# ============================================================================
# 1. READING DATA
# ============================================================================
print("\n[1] READING DATA...")
data_path = "APY.csv"
df = pd.read_csv(data_path)

print(f"✓ Data loaded successfully!")
print(f"  Total Records: {len(df):,}")
print(f"  Total Columns: {len(df.columns)}")

# Display schema
print("\n[SCHEMA]")
print(df.dtypes)

# Show sample data
print("\n[SAMPLE DATA]")
print(df.head())

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n[2] DATA EXPLORATION...")

# Check for missing values
print("\n[MISSING VALUES COUNT]")
print(df.isnull().sum())

# Statistical summary
print("\n[STATISTICAL SUMMARY]")
print(df.describe())

# ============================================================================
# 3. DATA CLEANING
# ============================================================================
print("\n[3] DATA CLEANING...")

# 3.1 Clean column names (remove extra spaces)
print("\n  [3.1] Cleaning column names...")
df.columns = df.columns.str.strip()

# 3.2 Handle missing values
print("\n  [3.2] Handling missing values...")
initial_count = len(df)

# Remove rows where critical columns are null
df = df.dropna(subset=['Yield', 'Production', 'Area', 'State', 'District', 'Crop', 'Season'])

final_count = len(df)
print(f"  ✓ Removed {initial_count - final_count:,} rows with missing critical values")
print(f"  ✓ Remaining records: {final_count:,}")

# 3.3 Remove duplicates
print("\n  [3.3] Removing duplicates...")
initial_count = len(df)
df = df.drop_duplicates()
final_count = len(df)
print(f"  ✓ Removed {initial_count - final_count:,} duplicate rows")
print(f"  ✓ Remaining records: {final_count:,}")

# 3.4 Data type conversions
print("\n  [3.4] Converting data types...")
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
df['Crop_Year'] = pd.to_numeric(df['Crop_Year'], errors='coerce').astype('Int64')

# Clean string columns
df['State'] = df['State'].str.strip()
df['District'] = df['District'].str.strip()
df['Crop'] = df['Crop'].str.strip()
df['Season'] = df['Season'].str.strip()

print("  ✓ Data types converted successfully")

# 3.5 Remove outliers
print("\n  [3.5] Handling outliers...")

# Filter out negative or zero yields
df = df[df['Yield'] > 0]
df = df[df['Area'] > 0]
df = df[df['Production'] > 0]

# Remove extreme outliers (Yield > 1000 is unrealistic for most crops)
df = df[df['Yield'] < 1000]

print(f"  ✓ Outliers handled. Remaining records: {len(df):,}")

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
df['Month'] = df['Season'].map(season_month_map)

print("  ✓ Month column created from Season")

# 4.2 Create productivity category
print("\n  [4.2] Creating productivity categories...")
df['Productivity_Category'] = pd.cut(
    df['Yield'],
    bins=[0, 1, 3, 10, float('inf')],
    labels=['Low', 'Medium', 'High', 'Very High']
)

print("  ✓ Productivity categories created")

# 4.3 Create decade column
print("\n  [4.3] Creating decade feature...")
df['Decade'] = (df['Crop_Year'] // 10) * 10

print("  ✓ Decade feature created")

# 4.4 Add unique ID
df['ID'] = range(len(df))

print("\n[FEATURE ENGINEERING COMPLETE]")
print(df.dtypes)

# ============================================================================
# 5. DATA ENCODING (for ML)
# ============================================================================
print("\n[5] PREPARING DATA FOR MACHINE LEARNING...")

# Create indexed columns for categorical variables
print("\n  [5.1] Encoding categorical variables...")

from sklearn.preprocessing import LabelEncoder

# Label Encoding
encoders = {}
for col in ['State', 'District', 'Crop', 'Season']:
    le = LabelEncoder()
    df[f'{col}_Index'] = le.fit_transform(df[col])
    encoders[col] = le

print("  ✓ Categorical encoding complete")

# ============================================================================
# 6. SAVE CLEANED DATA
# ============================================================================
print("\n[6] SAVING CLEANED DATA...")

# Create outputs directory if it doesn't exist
Path("outputs").mkdir(exist_ok=True)

# Save as CSV
csv_output_path = "outputs/cleaned_data.csv"
df.to_csv(csv_output_path, index=False)
print(f"  ✓ Data saved to: {csv_output_path}")

# Save as Parquet (efficient format)
parquet_output_path = "outputs/cleaned_data.parquet"
df.to_parquet(parquet_output_path, index=False)
print(f"  ✓ Data saved to: {parquet_output_path}")

# Save encoders
encoders_path = "outputs/encoders.json"
encoders_dict = {k: list(v.classes_) for k, v in encoders.items()}
with open(encoders_path, 'w') as f:
    json.dump(encoders_dict, f, indent=2)
print(f"  ✓ Encoders saved to: {encoders_path}")

# ============================================================================
# 7. DATA SUMMARY STATISTICS
# ============================================================================
print("\n[7] FINAL DATA SUMMARY...")

print("\n[RECORD COUNT BY CROP (Top 20)]")
print(df['Crop'].value_counts().head(20))

print("\n[RECORD COUNT BY STATE (Top 15)]")
print(df['State'].value_counts().head(15))

print("\n[RECORD COUNT BY SEASON]")
print(df['Season'].value_counts())

print("\n[AVERAGE YIELD BY CROP (Top 20)]")
print(df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(20))

print("\n[AVERAGE YIELD BY STATE (Top 15)]")
print(df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(15))

print("\n[YIELD STATISTICS BY SEASON]")
print(df.groupby('Season')['Yield'].agg(['mean', 'std', 'min', 'max']))

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFinal Dataset Statistics:")
print(f"  Total Records: {len(df):,}")
print(f"  Total Features: {len(df.columns)}")
print(f"  Unique Crops: {df['Crop'].nunique()}")
print(f"  Unique States: {df['State'].nunique()}")
print(f"  Unique Districts: {df['District'].nunique()}")
print(f"  Year Range: {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")
print("\n" + "=" * 80)
