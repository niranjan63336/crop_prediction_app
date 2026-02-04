"""
Simplified Model Training Script (Scikit-learn-based)
Crop Yield Prediction and Recommendation System
This version works without PySpark/Java
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

print("=" * 80)
print("CROP YIELD PREDICTION - MODEL TRAINING (SCIKIT-LEARN VERSION)")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] LOADING PREPROCESSED DATA...")

# Try to load from preprocessed file first
try:
    df = pd.read_csv("outputs/cleaned_data.csv")
    print(f"✓ Loaded preprocessed data")
except:
    print("  Preprocessed data not found. Loading and processing raw data...")
    df = pd.read_csv("APY.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Yield', 'Production', 'Area'])
    df = df[df['Yield'] > 0]
    
    # Add Month column
    season_month_map = {
        "Kharif": 7, "Rabi": 11, "Summer": 3,
        "Autumn": 9, "Whole Year": 1
    }
    df['Month'] = df['Season'].map(season_month_map)
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    for col in ['State', 'District', 'Crop', 'Season']:
        le = LabelEncoder()
        df[f'{col}_Index'] = le.fit_transform(df[col])

print(f"  Total Records: {len(df):,}")
print(f"  Total Features: {len(df.columns)}")

# ============================================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[2] PREPARING FEATURES AND TARGET...")

# Select features for the model
feature_columns = ['State_Index', 'District_Index', 'Crop_Index', 'Season_Index', 'Area', 'Crop_Year', 'Month']
target_column = 'Yield'

# Check if all feature columns exist
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"  Warning: Missing columns {missing_cols}. Creating them...")
    # If indices don't exist, create them
    from sklearn.preprocessing import LabelEncoder
    for col in ['State', 'District', 'Crop', 'Season']:
        if f'{col}_Index' not in df.columns:
            le = LabelEncoder()
            df[f'{col}_Index'] = le.fit_transform(df[col])

X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"  ✓ Features shape: {X.shape}")
print(f"  ✓ Target shape: {y.shape}")

# Handle any remaining missing values
X = X.fillna(0)
y = y.fillna(y.mean())

print("\n[FEATURE COLUMNS]")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3] SPLITTING DATA...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  ✓ Training set: {len(X_train):,} samples")
print(f"  ✓ Testing set: {len(X_test):,} samples")

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================
print("\n[4] TRAINING RANDOM FOREST MODEL...")
print("  This may take a few minutes...")

# Create and train the model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

model.fit(X_train, y_train)

print("  ✓ Model training completed!")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n[5] EVALUATING MODEL...")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n[MODEL PERFORMANCE METRICS]")
print("\nTraining Set:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"  R²:   {train_r2:.4f}")

print("\nTesting Set:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  R²:   {test_r2:.4f}")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n[6] FEATURE IMPORTANCE ANALYSIS...")

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n[FEATURE IMPORTANCE]")
print(feature_importance.to_string(index=False))

# ============================================================================
# 7. SAMPLE PREDICTIONS
# ============================================================================
print("\n[7] SAMPLE PREDICTIONS...")

# Show some sample predictions
sample_size = min(10, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)

print("\n[ACTUAL VS PREDICTED (Sample)]")
print(f"{'Actual':>10} {'Predicted':>10} {'Difference':>12}")
print("-" * 35)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred_test[idx]
    diff = actual - predicted
    print(f"{actual:>10.2f} {predicted:>10.2f} {diff:>12.2f}")

# ============================================================================
# 8. SAVE MODEL
# ============================================================================
print("\n[8] SAVING MODEL...")

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save the model
model_path = "models/yield_prediction_model.pkl"
joblib.dump(model, model_path)
print(f"  ✓ Model saved to: {model_path}")

# Save metrics
metrics = {
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse),
    "train_mae": float(train_mae),
    "test_mae": float(test_mae),
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "feature_importance": feature_importance.to_dict('records')
}

metrics_path = "models/model_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ Metrics saved to: {metrics_path}")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"  Algorithm: Random Forest Regressor")
print(f"  Number of Trees: 100")
print(f"  Max Depth: 10")
print(f"  Features Used: {len(feature_columns)}")
print(f"  Training Samples: {len(X_train):,}")
print(f"  Testing Samples: {len(X_test):,}")
print(f"\nPerformance:")
print(f"  Test R² Score: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Test MAE: {test_mae:.4f}")
print(f"\nModel saved to: {model_path}")
print("=" * 80)
