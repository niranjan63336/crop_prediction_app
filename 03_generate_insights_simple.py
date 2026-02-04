"""
Simplified Insight Generation Script (Pandas-based)
Crop Yield Prediction and Recommendation System
This version works without PySpark/Java
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("CROP YIELD PREDICTION - EXPLORATORY DATA ANALYSIS (PANDAS VERSION)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")

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

print(f"✓ Loaded {len(df):,} records")

# ============================================================================
# COMPREHENSIVE INSIGHTS GENERATION
# ============================================================================

insights = []

insights.append("=" * 80)
insights.append("CROP YIELD PREDICTION SYSTEM - COMPREHENSIVE INSIGHTS")
insights.append("=" * 80)
insights.append("")

# 1. DATASET OVERVIEW
insights.append("\n[1] DATASET OVERVIEW")
insights.append("-" * 80)
insights.append(f"Total Records: {len(df):,}")
insights.append(f"Total Features: {len(df.columns)}")
insights.append(f"Unique Crops: {df['Crop'].nunique()}")
insights.append(f"Unique States: {df['State'].nunique()}")
insights.append(f"Unique Districts: {df['District'].nunique()}")
insights.append(f"Unique Seasons: {df['Season'].nunique()}")
if 'Crop_Year' in df.columns:
    insights.append(f"Year Range: {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")

# 2. TOP PERFORMING CROPS
insights.append("\n\n[2] TOP PERFORMING CROPS")
insights.append("-" * 80)

# By Production
insights.append("\nTop 10 Crops by Total Production:")
top_production = df.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(10)

for idx, (crop, prod) in enumerate(top_production.items(), 1):
    insights.append(f"  {idx}. {crop}: {prod:,.2f} tonnes")

# By Average Yield
insights.append("\nTop 10 Crops by Average Yield:")
top_yield = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)

for idx, (crop, yld) in enumerate(top_yield.items(), 1):
    insights.append(f"  {idx}. {crop}: {yld:.2f} tonnes/hectare")

# 3. REGIONAL ANALYSIS
insights.append("\n\n[3] REGIONAL ANALYSIS")
insights.append("-" * 80)

# Top States by Area
insights.append("\nTop 10 States by Cultivated Area:")
top_states_area = df.groupby("State")["Area"].sum().sort_values(ascending=False).head(10)

for idx, (state, area) in enumerate(top_states_area.items(), 1):
    insights.append(f"  {idx}. {state}: {area:,.2f} hectares")

# Top States by Production
insights.append("\nTop 10 States by Total Production:")
top_states_prod = df.groupby("State")["Production"].sum().sort_values(ascending=False).head(10)

for idx, (state, prod) in enumerate(top_states_prod.items(), 1):
    insights.append(f"  {idx}. {state}: {prod:,.2f} tonnes")

# Top States by Average Yield
insights.append("\nTop 10 States by Average Yield:")
top_states_yield = df.groupby("State")["Yield"].mean().sort_values(ascending=False).head(10)

for idx, (state, yld) in enumerate(top_states_yield.items(), 1):
    insights.append(f"  {idx}. {state}: {yld:.2f} tonnes/hectare")

# 4. SEASONAL TRENDS
insights.append("\n\n[4] SEASONAL TRENDS")
insights.append("-" * 80)

seasonal_stats = df.groupby("Season").agg({
    "Yield": "mean",
    "Production": "sum",
    "Area": "sum"
}).sort_values("Yield", ascending=False)
seasonal_counts = df["Season"].value_counts()

insights.append("\nSeasonal Performance Analysis:")
for season in seasonal_stats.index:
    insights.append(f"\n{season}:")
    insights.append(f"  Average Yield: {seasonal_stats.loc[season, 'Yield']:.2f} tonnes/hectare")
    insights.append(f"  Total Production: {seasonal_stats.loc[season, 'Production']:,.2f} tonnes")
    insights.append(f"  Total Area: {seasonal_stats.loc[season, 'Area']:,.2f} hectares")
    insights.append(f"  Number of Records: {seasonal_counts[season]:,}")

# 5. TEMPORAL TRENDS
insights.append("\n\n[5] TEMPORAL TRENDS")
insights.append("-" * 80)

if 'Crop_Year' in df.columns:
    yearly_trends = df.groupby("Crop_Year").agg({
        "Yield": "mean",
        "Production": "sum",
        "Area": "sum"
    }).sort_index()

    # Get first and last year stats
    first_year_idx = yearly_trends.index[0]
    last_year_idx = yearly_trends.index[-1]
    
    first_year = yearly_trends.loc[first_year_idx]
    last_year = yearly_trends.loc[last_year_idx]

    insights.append(f"\nYear-over-Year Analysis:")
    insights.append(f"  First Year ({first_year_idx}):")
    insights.append(f"    Average Yield: {first_year['Yield']:.2f} tonnes/hectare")
    insights.append(f"    Total Production: {first_year['Production']:,.2f} tonnes")
    insights.append(f"\n  Latest Year ({last_year_idx}):")
    insights.append(f"    Average Yield: {last_year['Yield']:.2f} tonnes/hectare")
    insights.append(f"    Total Production: {last_year['Production']:,.2f} tonnes")

    yield_change = ((last_year['Yield'] - first_year['Yield']) / first_year['Yield']) * 100
    insights.append(f"\n  Yield Change: {yield_change:+.2f}%")

# 6. CROP-WISE INSIGHTS
insights.append("\n\n[6] CROP-WISE DETAILED INSIGHTS")
insights.append("-" * 80)

# Most widely cultivated crops
insights.append("\nMost Widely Cultivated Crops (by area):")
wide_crops = df.groupby("Crop")["Area"].sum().sort_values(ascending=False).head(10)

for idx, (crop, area) in enumerate(wide_crops.items(), 1):
    insights.append(f"  {idx}. {crop}: {area:,.2f} hectares")

# Most productive crops (total production)
insights.append("\nMost Productive Crops (by total output):")
productive_crops = df.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(10)

for idx, (crop, prod) in enumerate(productive_crops.items(), 1):
    insights.append(f"  {idx}. {crop}: {prod:,.2f} tonnes")

# 7. STATE-CROP COMBINATIONS
insights.append("\n\n[7] BEST STATE-CROP COMBINATIONS")
insights.append("-" * 80)

state_crop_yield = df.groupby(["State", "Crop"])["Yield"].mean().sort_values(ascending=False).head(15)

insights.append("\nTop 15 State-Crop Combinations by Yield:")
for idx, ((state, crop), yld) in enumerate(state_crop_yield.items(), 1):
    insights.append(f"  {idx}. {state} - {crop}: {yld:.2f} tonnes/hectare")

# 8. PRODUCTIVITY ANALYSIS
insights.append("\n\n[8] PRODUCTIVITY ANALYSIS")
insights.append("-" * 80)

# Overall statistics
insights.append("\nOverall Productivity Statistics:")
insights.append(f"  Mean Yield: {df['Yield'].mean():.2f} tonnes/hectare")
insights.append(f"  Standard Deviation: {df['Yield'].std():.2f}")
insights.append(f"  Minimum Yield: {df['Yield'].min():.2f} tonnes/hectare")
insights.append(f"  Maximum Yield: {df['Yield'].max():.2f} tonnes/hectare")
insights.append(f"  Mean Area: {df['Area'].mean():.2f} hectares")
insights.append(f"  Mean Production: {df['Production'].mean():.2f} tonnes")

# 9. KEY RECOMMENDATIONS
insights.append("\n\n[9] KEY RECOMMENDATIONS FOR FARMERS")
insights.append("-" * 80)

insights.append("""
Based on comprehensive analysis of agricultural data:

1. CROP SELECTION:
   - Focus on high-yield crops proven in your region
   - Consider crops with consistent performance across seasons
   - Diversify to reduce risk and optimize returns

2. SEASONAL PLANNING:
   - Align planting schedules with optimal seasons for your crops
   - Monitor seasonal yield patterns for better timing
   - Plan irrigation and resource allocation accordingly

3. REGIONAL BEST PRACTICES:
   - Study successful state-crop combinations
   - Adopt techniques from high-performing regions
   - Adapt practices to local conditions

4. PRODUCTIVITY OPTIMIZATION:
   - Target yields above regional averages
   - Invest in soil health and proper fertilization
   - Implement efficient water management

5. MARKET STRATEGY:
   - Focus on crops with high total production (market demand)
   - Balance between yield and market prices
   - Consider value-added opportunities

6. RISK MANAGEMENT:
   - Diversify crop portfolio
   - Use historical data for planning
   - Monitor weather patterns and adapt

7. SUSTAINABLE PRACTICES:
   - Practice crop rotation
   - Maintain soil fertility
   - Optimize resource utilization
""")

# 10. SURPRISING OBSERVATIONS
insights.append("\n\n[10] SURPRISING OBSERVATIONS")
insights.append("-" * 80)

# Find crops with high yield but low total area
overall_mean_yield = df['Yield'].mean()
crop_stats = df.groupby("Crop").agg({
    "Yield": "mean",
    "Area": "sum"
})

high_yield_low_area = crop_stats[crop_stats["Yield"] > overall_mean_yield].sort_values("Area").head(5)

insights.append("\nHigh-Yield Underutilized Crops (Opportunities):")
for idx, (crop, row) in enumerate(high_yield_low_area.iterrows(), 1):
    insights.append(f"  {idx}. {crop}: Yield {row['Yield']:.2f}, Area {row['Area']:,.2f} hectares")
insights.append("\n  → These crops show high yields but are cultivated in smaller areas.")
insights.append("  → Potential opportunity for expansion and increased production.")

# 11. TRENDS AND PATTERNS
insights.append("\n\n[11] IDENTIFIED TRENDS AND PATTERNS")
insights.append("-" * 80)

insights.append("""
1. YIELD TRENDS:
   - Certain crops consistently show higher yields across regions
   - Seasonal variations significantly impact crop performance
   - Modern agricultural practices have improved yields over time

2. REGIONAL PATTERNS:
   - Some states specialize in specific crops with exceptional yields
   - Climate and soil conditions create regional advantages
   - Infrastructure and technology adoption vary by region

3. SEASONAL PATTERNS:
   - Monsoon (Kharif) and Winter (Rabi) seasons dominate production
   - Seasonal crops show distinct yield patterns
   - Multi-season crops provide year-round opportunities

4. PRODUCTION PATTERNS:
   - Staple crops dominate total production volume
   - Cash crops often show higher yields per hectare
   - Regional specialization drives production efficiency
""")

# 12. CONCLUSION
insights.append("\n\n[12] CONCLUSION")
insights.append("-" * 80)

insights.append(f"""
This comprehensive analysis of {len(df):,} agricultural records reveals:

✓ Significant variation in crop performance across regions and seasons
✓ Clear opportunities for yield optimization through data-driven decisions
✓ Importance of matching crops to regional and seasonal conditions
✓ Potential for expanding high-yield underutilized crops
✓ Value of learning from top-performing state-crop combinations

The Crop Yield Prediction and Recommendation System leverages these insights
to provide farmers with actionable, data-driven recommendations for:
- Optimal crop selection
- Seasonal planning
- Yield maximization
- Risk mitigation
- Sustainable agricultural practices

By combining Big Data analytics with machine learning, this system empowers
farmers to make informed decisions that can significantly improve agricultural
productivity and profitability.
""")

insights.append("\n" + "=" * 80)
insights.append("END OF INSIGHTS REPORT")
insights.append("=" * 80)

# Save insights to file
print("\n[SAVING INSIGHTS...]")
insights_text = "\n".join(insights)

# Create outputs directory if needed
Path("outputs").mkdir(exist_ok=True)

with open("outputs/insights.txt", "w", encoding='utf-8') as f:
    f.write(insights_text)

print("✓ Insights saved to: outputs/insights.txt")

print("\n✓ EDA COMPLETED SUCCESSFULLY!")
