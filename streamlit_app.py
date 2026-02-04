"""
Crop Yield Prediction and Recommendation System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1 {
        color: #2c5f2d;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #3a7d44;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    try:
        # Try to load from CSV first
        csv_files = list(Path("outputs/cleaned_data.csv").parent.glob("*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
        else:
            # Fallback to original data
            df = pd.read_csv("APY.csv")
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Basic cleaning
            df = df.dropna(subset=['Yield', 'Production', 'Area'])
            df = df[df['Yield'] > 0]
            
            # Add Month column based on Season
            season_month_map = {
                "Kharif": 7, "Rabi": 11, "Summer": 3, 
                "Autumn": 9, "Whole Year": 1
            }
            df['Month'] = df['Season'].map(season_month_map)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🌾 Crop Analytics")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🌾 Crop Recommendation", "📊 Visualizations", "💡 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **About this App**
    
    This Big Data Analytics application helps farmers:
    - Predict crop yields
    - Get crop recommendations
    - Analyze agricultural trends
    - Make data-driven decisions
""")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    # Header
    st.title("🌾 Crop Yield Prediction & Recommendation System")
    st.markdown("### *Empowering Farmers with Data-Driven Insights*")
    st.markdown("---")
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>📊</h2>
                <h3 style='color: white;'>Big Data Analytics</h3>
                <p>Processing 345K+ agricultural records</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>🤖</h2>
                <h3 style='color: white;'>ML Predictions</h3>
                <p>Random Forest algorithm for accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>🌱</h2>
                <h3 style='color: white;'>Smart Recommendations</h3>
                <p>Optimized crop selection system</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("## 📈 Dataset Overview")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Crops", f"{df['Crop'].nunique()}")
        with col3:
            st.metric("States Covered", f"{df['State'].nunique()}")
        with col4:
            st.metric("Year Range", f"{df['Crop_Year'].min()}-{df['Crop_Year'].max()}")
        
        # Quick stats
        st.markdown("### 📊 Quick Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Crops by Production**")
            top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_crops.values,
                y=top_crops.index,
                orientation='h',
                labels={'x': 'Total Production', 'y': 'Crop'},
                color=top_crops.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 States by Area**")
            top_states = df.groupby('State')['Area'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_states.values,
                y=top_states.index,
                orientation='h',
                labels={'x': 'Total Area (hectares)', 'y': 'State'},
                color=top_states.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Features section
    st.markdown("---")
    st.markdown("## 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌾 Crop Recommendation
        - Select your state, month, and land area
        - Get personalized crop recommendations
        - View predicted yields
        - Optimize your agricultural decisions
        """)
        
        st.markdown("""
        ### 📊 Data Visualizations
        - Interactive charts and graphs
        - Correlation heatmaps
        - Trend analysis
        - Distribution plots
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Advanced Analytics
        - PySpark-powered big data processing
        - Machine learning predictions
        - Feature importance analysis
        - Statistical insights
        """)
        
        st.markdown("""
        ### 💡 Actionable Insights
        - Regional crop performance
        - Seasonal trends
        - Yield optimization tips
        - Data-driven recommendations
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #e8f5e9; border-radius: 10px;'>
            <h3 style='color: #2c5f2d;'>Ready to Get Started?</h3>
            <p style='font-size: 18px;'>Use the sidebar to navigate to Crop Recommendation or Visualizations!</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# CROP RECOMMENDATION PAGE
# ============================================================================
elif page == "🌾 Crop Recommendation":
    st.title("🌾 Crop Recommendation System")
    st.markdown("### Get personalized crop recommendations based on your location and land area")
    st.markdown("---")
    
    if df is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Enter Your Details")
            
            # State selection
            states = sorted(df['State'].unique())
            selected_state = st.selectbox("🗺️ Select State", states)
            
            # Month selection
            months = {
                1: "January (Whole Year)", 3: "March (Summer)", 
                7: "July (Kharif/Monsoon)", 9: "September (Autumn)", 
                11: "November (Rabi/Winter)"
            }
            selected_month = st.selectbox(
                "📅 Select Month/Season",
                options=list(months.keys()),
                format_func=lambda x: months[x]
            )
            
            # Area input
            area = st.number_input(
                "📏 Enter Land Area (in hectares)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0,
                step=0.5
            )
            
            # Recommend button
            if st.button("🔍 Get Recommendation", use_container_width=True):
                # Try exact match first
                filtered_df = df[
                    (df['State'] == selected_state) &
                    (df['Month'] == selected_month)
                ]
                
                # If no exact match, try state only
                if len(filtered_df) == 0:
                    st.warning(f"No data for {selected_state} in selected month. Using state-wide data...")
                    filtered_df = df[df['State'] == selected_state]
                
                # If still no data, use month/season data from all states
                if len(filtered_df) == 0:
                    st.warning(f"No data for {selected_state}. Using national data for selected season...")
                    filtered_df = df[df['Month'] == selected_month]
                
                # If still no data, use all data
                if len(filtered_df) == 0:
                    st.warning("Using overall dataset for recommendations...")
                    filtered_df = df
                
                if len(filtered_df) > 0:
                    # Calculate average yield per crop
                    crop_performance = filtered_df.groupby('Crop').agg({
                        'Yield': 'mean',
                        'Production': 'mean',
                        'Area': 'mean'
                    }).reset_index()
                    
                    crop_performance = crop_performance.sort_values('Yield', ascending=False)
                    
                    # Get top recommendation
                    best_crop = crop_performance.iloc[0]
                    predicted_yield = best_crop['Yield']
                    predicted_production = predicted_yield * area
                    
                    st.session_state['recommendation'] = {
                        'crop': best_crop['Crop'],
                        'yield': predicted_yield,
                        'production': predicted_production,
                        'area': area,
                        'top_crops': crop_performance.head(5),
                        'data_source': 'exact' if (df['State'] == selected_state).any() and (df['Month'] == selected_month).any() else 'fallback'
                    }
                else:
                    st.error("Unable to generate recommendations. Please check the dataset.")
        
        with col2:
            st.markdown("### 🎯 Recommendation Results")
            
            if 'recommendation' in st.session_state:
                rec = st.session_state['recommendation']
                
                # Display recommendation
                st.success(f"**Recommended Crop: {rec['crop']}** 🌱")
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Expected Yield", f"{rec['yield']:.2f} tonnes/hectare")
                with col_b:
                    st.metric("Expected Production", f"{rec['production']:.2f} tonnes")
                
                st.markdown("---")
                
                # Top 5 crops
                st.markdown("### 📊 Top 5 Recommended Crops")
                
                top_crops_df = rec['top_crops'][['Crop', 'Yield']].copy()
                top_crops_df['Yield'] = top_crops_df['Yield'].round(2)
                top_crops_df.columns = ['Crop Name', 'Avg Yield (tonnes/hectare)']
                top_crops_df.index = range(1, len(top_crops_df) + 1)
                
                st.dataframe(top_crops_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    rec['top_crops'].head(5),
                    x='Crop',
                    y='Yield',
                    title='Yield Comparison of Top 5 Crops',
                    labels={'Yield': 'Average Yield (tonnes/hectare)', 'Crop': 'Crop Name'},
                    color='Yield',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Fill in your details and click 'Get Recommendation' to see results")
                
                # Show sample data
                st.markdown("### 📋 Sample Data Preview")
                sample_data = df[df['State'] == selected_state].head(10)
                st.dataframe(sample_data[['Crop', 'Season', 'Area', 'Yield']], use_container_width=True)

# ============================================================================
# VISUALIZATIONS PAGE
# ============================================================================
elif page == "📊 Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("### Interactive visualizations and insights")
    st.markdown("---")
    
    if df is not None:
        # Sidebar for plot selection
        st.sidebar.markdown("### 📈 Select Visualization")
        plot_type = st.sidebar.selectbox(
            "Choose a plot:",
            ["Histogram", "Bar Chart", "Pie Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
        )
        
        # Main content area
        if plot_type == "Histogram":
            st.markdown("## 📊 Histogram - Distribution Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                numeric_cols = ['Area', 'Production', 'Yield', 'Crop_Year']
                selected_col = st.selectbox("Select variable:", numeric_cols)
                
                fig = px.histogram(
                    df,
                    x=selected_col,
                    nbins=50,
                    title=f'Distribution of {selected_col}',
                    labels={selected_col: selected_col},
                    color_discrete_sequence=['#2ecc71']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.info(f"""
                **{selected_col} Distribution:**
                
                - Mean: {df[selected_col].mean():.2f}
                - Median: {df[selected_col].median():.2f}
                - Std Dev: {df[selected_col].std():.2f}
                - Min: {df[selected_col].min():.2f}
                - Max: {df[selected_col].max():.2f}
                
                This histogram shows the frequency distribution of {selected_col} values across the dataset.
                """)
        
        elif plot_type == "Bar Chart":
            st.markdown("## 📊 Bar Chart - Category Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                category = st.selectbox("Select category:", ['Crop', 'State', 'Season'])
                metric = st.selectbox("Select metric:", ['Production', 'Yield', 'Area'])
                top_n = st.slider("Show top N:", 5, 20, 10)
                
                grouped_data = df.groupby(category)[metric].mean().sort_values(ascending=False).head(top_n)
                
                fig = px.bar(
                    x=grouped_data.index,
                    y=grouped_data.values,
                    title=f'Top {top_n} {category} by Average {metric}',
                    labels={'x': category, 'y': f'Average {metric}'},
                    color=grouped_data.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.info(f"""
                **Top {category} Analysis:**
                
                This bar chart shows the top {top_n} {category} ranked by average {metric}.
                
                **Key Insights:**
                - Highest: {grouped_data.index[0]} ({grouped_data.values[0]:.2f})
                - Lowest in top {top_n}: {grouped_data.index[-1]} ({grouped_data.values[-1]:.2f})
                - Range: {grouped_data.values[0] - grouped_data.values[-1]:.2f}
                """)
        
        elif plot_type == "Pie Chart":
            st.markdown("## 📊 Pie Chart - Proportion Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                category = st.selectbox("Select category:", ['Crop', 'State', 'Season'])
                metric = st.selectbox("Select metric:", ['Production', 'Area'])
                top_n = st.slider("Show top N:", 5, 15, 10)
                
                grouped_data = df.groupby(category)[metric].sum().sort_values(ascending=False).head(top_n)
                
                fig = px.pie(
                    values=grouped_data.values,
                    names=grouped_data.index,
                    title=f'Top {top_n} {category} by Total {metric}',
                    hole=0.3  # Makes it a donut chart
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                total = grouped_data.sum()
                top_percentage = (grouped_data.values[0] / total) * 100
                st.info(f"""
                **Proportion Analysis:**
                
                This pie chart shows the distribution of {metric} across different {category}.
                
                **Key Insights:**
                - Top contributor: {grouped_data.index[0]} ({top_percentage:.1f}%)
                - Total {metric}: {total:,.2f}
                - Number of categories shown: {len(grouped_data)}
                
                The size of each slice represents the proportion of total {metric}.
                """)
        
        
        elif plot_type == "Scatter Plot":
            st.markdown("## 📊 Scatter Plot - Relationship Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                x_var = st.selectbox("Select X variable:", ['Area', 'Production', 'Crop_Year'])
                y_var = st.selectbox("Select Y variable:", ['Yield', 'Production', 'Area'])
                
                # Sample data for performance
                sample_df = df.sample(min(5000, len(df)))
                
                fig = px.scatter(
                    sample_df,
                    x=x_var,
                    y=y_var,
                    color='Season',
                    title=f'{y_var} vs {x_var}',
                    labels={x_var: x_var, y_var: y_var},
                    opacity=0.6
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                correlation = df[[x_var, y_var]].corr().iloc[0, 1]
                st.info(f"""
                **Relationship Analysis:**
                
                Correlation coefficient: {correlation:.3f}
                
                **Interpretation:**
                {
                    "Strong positive correlation" if correlation > 0.7 else
                    "Moderate positive correlation" if correlation > 0.3 else
                    "Weak positive correlation" if correlation > 0 else
                    "Weak negative correlation" if correlation > -0.3 else
                    "Moderate negative correlation" if correlation > -0.7 else
                    "Strong negative correlation"
                }
                
                This scatter plot shows the relationship between {x_var} and {y_var}.
                """)
        
        elif plot_type == "Box Plot":
            st.markdown("## 📊 Box Plot - Distribution by Category")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                category = st.selectbox("Select category:", ['Season', 'Crop'])
                metric = st.selectbox("Select metric:", ['Yield', 'Production', 'Area'])
                
                # Filter top categories for better visualization
                if category == 'Crop':
                    top_categories = df.groupby(category)[metric].mean().sort_values(ascending=False).head(10).index
                    plot_df = df[df[category].isin(top_categories)]
                else:
                    plot_df = df
                
                fig = px.box(
                    plot_df,
                    x=category,
                    y=metric,
                    title=f'{metric} Distribution by {category}',
                    color=category
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.info(f"""
                **Box Plot Analysis:**
                
                This visualization shows the distribution of {metric} across different {category} values.
                
                **Key Elements:**
                - Box: Interquartile range (IQR)
                - Line in box: Median
                - Whiskers: Data range
                - Points: Outliers
                
                Use this to identify variations and outliers across categories.
                """)
        
        elif plot_type == "Correlation Heatmap":
            st.markdown("## 📊 Correlation Heatmap")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Select numeric columns
                numeric_df = df[['Area', 'Production', 'Yield', 'Crop_Year']].copy()
                if 'Month' in df.columns:
                    numeric_df['Month'] = df['Month']
                
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    title='Correlation Matrix of Numeric Variables',
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.info("""
                **Correlation Heatmap:**
                
                This heatmap shows correlations between numeric variables.
                
                **Color Scale:**
                - Red: Negative correlation
                - White: No correlation
                - Blue: Positive correlation
                
                **Values:**
                - +1: Perfect positive correlation
                - 0: No correlation
                - -1: Perfect negative correlation
                
                Strong correlations indicate variables that move together.
                """)

# ============================================================================
# INSIGHTS PAGE
# ============================================================================
elif page == "💡 Insights":
    st.title("💡 Key Insights & Recommendations")
    st.markdown("### Data-driven insights from agricultural analysis")
    st.markdown("---")
    
    if df is not None:
        # Top insights
        st.markdown("## 🔍 Key Findings")
        
        # Calculate insights
        top_crop = df.groupby('Crop')['Production'].sum().idxmax()
        top_state = df.groupby('State')['Area'].sum().idxmax()
        best_season = df.groupby('Season')['Yield'].mean().idxmax()
        avg_yield = df['Yield'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌾 Crop Performance
            """)
            st.success(f"""
            **Highest Production Crop:** {top_crop}
            
            This crop shows the highest total production across all regions and years in the dataset.
            """)
            
            st.markdown("""
            ### 📍 Regional Analysis
            """)
            st.info(f"""
            **Largest Agricultural State:** {top_state}
            
            This state has the maximum total cultivated area, indicating significant agricultural activity.
            """)
        
        with col2:
            st.markdown("""
            ### 📅 Seasonal Trends
            """)
            st.success(f"""
            **Best Season for Yield:** {best_season}
            
            Crops grown in this season show the highest average yield across the dataset.
            """)
            
            st.markdown("""
            ### 📊 Overall Performance
            """)
            st.info(f"""
            **Average Yield:** {avg_yield:.2f} tonnes/hectare
            
            This represents the overall agricultural productivity across all crops and regions.
            """)
        
        st.markdown("---")
        
        # Detailed insights
        st.markdown("## 📈 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🌾 Crop Insights", "🗺️ Regional Insights", "📅 Temporal Trends"])
        
        with tab1:
            st.markdown("### Top Performing Crops")
            
            crop_stats = df.groupby('Crop').agg({
                'Yield': 'mean',
                'Production': 'sum',
                'Area': 'sum'
            }).round(2)
            crop_stats = crop_stats.sort_values('Production', ascending=False).head(10)
            
            st.dataframe(crop_stats, use_container_width=True)
            
            st.markdown("""
            **Insights:**
            - Crops with high production indicate market demand and farmer preference
            - High yield crops are more efficient in terms of land utilization
            - Consider both yield and total production for crop selection
            """)
        
        with tab2:
            st.markdown("### Regional Performance")
            
            state_stats = df.groupby('State').agg({
                'Yield': 'mean',
                'Production': 'sum',
                'Area': 'sum'
            }).round(2)
            state_stats = state_stats.sort_values('Production', ascending=False).head(10)
            
            st.dataframe(state_stats, use_container_width=True)
            
            st.markdown("""
            **Insights:**
            - States with larger areas don't always have higher yields
            - Regional climate and soil conditions significantly impact productivity
            - Best practices from high-yield states can be adopted elsewhere
            """)
        
        with tab3:
            st.markdown("### Year-over-Year Trends")
            
            yearly_stats = df.groupby('Crop_Year').agg({
                'Yield': 'mean',
                'Production': 'sum',
                'Area': 'sum'
            }).round(2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_stats.index,
                y=yearly_stats['Yield'],
                mode='lines+markers',
                name='Average Yield',
                line=dict(color='#2ecc71', width=3)
            ))
            fig.update_layout(
                title='Average Yield Trend Over Years',
                xaxis_title='Year',
                yaxis_title='Average Yield',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insights:**
            - Identify years with exceptional or poor performance
            - Correlate trends with weather patterns and policies
            - Use historical trends for future planning
            """)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("## 🎯 Recommendations for Farmers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Best Practices
            
            1. **Crop Selection**
               - Choose crops with proven high yields in your region
               - Consider market demand and pricing
               - Diversify to reduce risk
            
            2. **Seasonal Planning**
               - Align planting with optimal seasons
               - Monitor weather forecasts
               - Plan for irrigation needs
            
            3. **Land Management**
               - Optimize land utilization
               - Practice crop rotation
               - Maintain soil health
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Risk Mitigation
            
            1. **Data-Driven Decisions**
               - Use historical data for planning
               - Monitor yield trends
               - Adapt to changing patterns
            
            2. **Resource Optimization**
               - Efficient water usage
               - Appropriate fertilizer application
               - Pest management
            
            3. **Market Strategy**
               - Understand demand patterns
               - Plan harvest timing
               - Explore value addition
            """)
        
        st.markdown("---")
        
        st.success("""
        ### 🌟 Final Recommendations
        
        Based on comprehensive analysis of 345,000+ agricultural records:
        
        - **Focus on high-yield crops** suitable for your region and season
        - **Leverage seasonal patterns** to maximize productivity
        - **Learn from top-performing regions** and adopt their best practices
        - **Use data analytics** for informed decision-making
        - **Diversify crops** to minimize risk and optimize returns
        - **Monitor trends** regularly to adapt to changing conditions
        
        This system provides evidence-based recommendations to help you make better agricultural decisions!
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌾 Crop Yield Prediction & Recommendation System | Built with PySpark & Streamlit</p>
        <p>Big Data Analytics Project | 2024</p>
    </div>
""", unsafe_allow_html=True)
