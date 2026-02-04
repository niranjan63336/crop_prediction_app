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

# Custom CSS for styling
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

# =======================================================================
# LOAD CLEANED DATA (UPDATED)
# =======================================================================
@st.cache_data
def load_data():
    """Load the cleaned dataset (PARQUET VERSION)"""
    try:
        df = pd.read_parquet("outputs/cleaned_data.parquet")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load data
df = load_data()

# =======================================================================
# SIDEBAR
# =======================================================================
st.sidebar.title("🌾 Crop Analytics")
st.sidebar.markdown("---")

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

# =======================================================================
# HOME PAGE
# =======================================================================
if page == "🏠 Home":
    st.title("🌾 Crop Yield Prediction & Recommendation System")
    st.markdown("### *Empowering Farmers with Data-Driven Insights*")
    st.markdown("---")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>📊</h2>
                    <h3>Big Data Analytics</h3>
                    <p>Processing 345K+ agricultural records</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>🤖</h2>
                    <h3>ML Predictions</h3>
                    <p>Random Forest algorithm for accuracy</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>🌱</h2>
                    <h3>Smart Recommendations</h3>
                    <p>Optimized crop selection system</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview
        st.markdown("## 📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Crops", f"{df['Crop'].nunique()}")
        with col3:
            st.metric("States Covered", f"{df['State'].nunique()}")
        with col4:
            st.metric("Year Range", f"{df['Crop_Year'].min()}-{df['Crop_Year'].max()}")

        st.markdown("### 📊 Quick Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Crops by Production**")
            top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_crops.values,
                y=top_crops.index,
                orientation='h',
                labels={'x': 'Production', 'y': 'Crop'},
                color=top_crops.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top 10 States by Area**")
            top_states = df.groupby('State')['Area'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_states.values,
                y=top_states.index,
                orientation='h',
                labels={'x': 'Area', 'y': 'State'},
                color=top_states.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

# =======================================================================
# CROP RECOMMENDATION PAGE
# =======================================================================
elif page == "🌾 Crop Recommendation":

    st.title("🌾 Crop Recommendation System")
    st.markdown("### Get personalized crop recommendations based on your location and land area")
    st.markdown("---")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Enter Your Details")

            states = sorted(df['State'].unique())
            selected_state = st.selectbox("🗺️ Select State", states)

            months = {
                1: "January (Whole Year)", 3: "March (Summer)",
                7: "July (Kharif)", 9: "September (Autumn)",
                11: "November (Rabi)"
            }
            selected_month = st.selectbox(
                "📅 Select Month/Season",
                options=list(months.keys()),
                format_func=lambda x: months[x]
            )

            area = st.number_input(
                "📏 Enter Land Area (hectares)", min_value=0.1, max_value=10000.0, value=10.0
            )

            if st.button("🔍 Get Recommendation", use_container_width=True):

                filtered_df = df[
                    (df['State'] == selected_state) &
                    (df['Month'] == selected_month)
                ]

                if filtered_df.empty:
                    filtered_df = df[df['State'] == selected_state]

                if filtered_df.empty:
                    filtered_df = df[df['Month'] == selected_month]

                if filtered_df.empty:
                    filtered_df = df

                crop_perf = filtered_df.groupby('Crop').agg({
                    'Yield': 'mean',
                    'Production': 'mean',
                    'Area': 'mean'
                }).reset_index()

                crop_perf = crop_perf.sort_values('Yield', ascending=False)

                best = crop_perf.iloc[0]
                predicted_yield = best['Yield']
                predicted_production = predicted_yield * area

                st.session_state['rec'] = {
                    'crop': best['Crop'],
                    'yield': predicted_yield,
                    'production': predicted_production,
                    'area': area,
                    'top': crop_perf.head(5)
                }

        with col2:
            if 'rec' in st.session_state:
                rec = st.session_state['rec']
                st.success(f"**Recommended Crop: {rec['crop']}** 🌱")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Expected Yield", f"{rec['yield']:.2f} t/ha")
                with col_b:
                    st.metric("Total Production", f"{rec['production']:.2f} t")

                top_df = rec['top'][['Crop', 'Yield']]
                top_df['Yield'] = top_df['Yield'].round(2)
                top_df.columns = ['Crop', 'Avg Yield']
                st.dataframe(top_df, use_container_width=True)

# =======================================================================
# VISUALIZATIONS PAGE (unchanged)
# =======================================================================

elif page == "📊 Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("### Interactive visualizations and insights")
    st.markdown("---")

    # (KEEP YOUR ORIGINAL VISUALIZATION CODE HERE — SAME AS BEFORE)

# =======================================================================
# INSIGHTS PAGE (unchanged)
# =======================================================================

elif page == "💡 Insights":
    st.title("💡 Key Insights & Recommendations")
    st.markdown("### Data-driven insights from agricultural analysis")
    st.markdown("---")

    # (KEEP YOUR ORIGINAL INSIGHTS CODE HERE — SAME AS BEFORE)

# FOOTER
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌾 Crop Yield Prediction & Recommendation System | Built with Streamlit</p>
        <p>Big Data Analytics Project | 2024</p>
    </div>
""", unsafe_allow_html=True)

