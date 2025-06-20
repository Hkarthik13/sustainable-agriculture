import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Agriculture Advisor | India",
    page_icon="🌿",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    /* Main background */
    .main {background-color: #F0F2F6;}

    /* Custom Cards for Metrics */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 7px solid #FF9933; /* Saffron color border */
        margin-bottom: 10px;
        text-align: center;
    }
    .metric-card h3 {
        color: #138808; /* Green color for title */
        font-size: 18px;
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .metric-card span {
        font-size: 16px;
        color: #555;
    }

    /* Page Headers */
    .header {color: #138808; font-size: 36px; font-weight: bold; text-align: center;}
    .subheader {color: #1B5E20; font-size: 26px; font-weight: bold; border-bottom: 2px solid #FF9933; padding-bottom: 5px; margin-top: 20px;}
    .caption-text {font-size: 14px; color: #555; text-align: center;}
    </style>
""", unsafe_allow_html=True)


# --- DATA LOADING & MODEL TRAINING (CACHED) ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("farmer_advisor_dataset.csv")
        return data
    except FileNotFoundError:
        st.error("FATAL: 'farmer_advisor_dataset.csv' not found. Please ensure it's in the same directory.")
        return None

@st.cache_data
def preprocess_data(_data):
    df = _data.copy().dropna()
    le = LabelEncoder().fit(['Wheat', 'Soybean', 'Corn', 'Rice'])
    df['Crop_Type_Encoded'] = le.transform(df['Crop_Type'])
    features = ['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Crop_Type_Encoded']
    X, y = df[features], df['Crop_Yield_ton']
    return X, y, le

@st.cache_resource
def train_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=10, min_samples_leaf=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model, mae, r2


# --- CORE RECOMMENDATION LOGIC ---
def generate_recommendations(inputs, model, le):
    all_recs = []
    crop_types = ['Wheat', 'Soybean', 'Corn', 'Rice']

    for crop in crop_types:
        crop_encoded = le.transform([crop])[0]
        input_data = np.array([[inputs['ph'], inputs['moisture'], inputs['temp'], inputs['rain'], crop_encoded]])
        yield_pred = model.predict(input_data)[0]

        # Calculate Sustainability Score
        optimal = {
            'Wheat': {'pH': (6.0, 7.5), 'Moisture': (20, 35), 'Temp': (15, 25), 'Rain': (50, 150)},
            'Soybean': {'pH': (6.2, 7.0), 'Moisture': (25, 40), 'Temp': (20, 30), 'Rain': (100, 200)},
            'Corn': {'pH': (5.8, 7.0), 'Moisture': (25, 40), 'Temp': (20, 30), 'Rain': (100, 200)},
            'Rice': {'pH': (5.5, 6.5), 'Moisture': (30, 50), 'Temp': (20, 35), 'Rain': (150, 300)}
        }
        scores, advice_list = [], []
        for param, (low, high) in optimal[crop].items():
            val = inputs[param.lower()]
            if low <= val <= high:
                scores.append(100)
                advice_list.append(f"✅ {param} ({val:.1f}) is within the optimal range of {low}-{high}.")
            else:
                deviation = min(abs(val - low), abs(val - high))
                score = max(0, 100 - (deviation / ((high - low) / 2)) * 100)
                scores.append(score)
                advice_list.append(f"⚠️ {param} ({val:.1f}) is outside the optimal range of {low}-{high}. Adjustment needed.")
        sustainability_score = np.mean(scores)

        # Calculate Profitability in Rupees
        profit = (yield_pred * inputs['price']) - inputs['cost']

        # Other Details
        hemisphere = 'Northern' if 'north' in inputs['loc'].lower() else 'Southern'
        planting_times = {'Wheat': ('Rabi (Oct-Nov)', 'Winter'), 'Soybean': ('Kharif (Jun-Jul)', 'Summer'), 'Corn': ('Kharif (Jun-Jul)', 'Summer'), 'Rice': ('Kharif (Jun-Jul)', 'Summer')}
        planting_time = planting_times[crop][0] if hemisphere == 'Northern' else planting_times[crop][1]
        
        # High-quality images for guaranteed display
        crop_images = {
            'Wheat': 'https://images.unsplash.com/photo-1542282246-53872195a613?w=500',
            'Soybean': 'https://images.unsplash.com/photo-1620912282585-781e42b26c61?w=500',
            'Corn': 'https://images.unsplash.com/photo-1598164344933-a868a2d131f1?w=500',
            'Rice': 'https://images.unsplash.com/photo-1536384459922-b9b7617c4a16?w=500'
        }

        all_recs.append({
            "Crop": crop, "Yield": yield_pred, "Profit": profit, "Sustainability": sustainability_score,
            "Image": crop_images[crop], "Planting Time": planting_time, "Advice": advice_list
        })
    return sorted(all_recs, key=lambda x: x['Profit'], reverse=True)


# --- UI HELPER FUNCTIONS ---
def create_metric_card(title, value, unit, icon):
    st.markdown(f"""
        <div class="metric-card">
            <h3>{icon} {title}</h3>
            <p>{value}</p>
            <span>{unit}</span>
        </div>
    """, unsafe_allow_html=True)

# --- INITIALIZE APP ---
data = load_data()
if data is None:
    st.stop()
X, y, le = preprocess_data(data)
model, mae, r2 = train_model(X, y)

# --- MAIN APP LAYOUT ---
st.markdown("<div class='header'>🇮🇳 Advanced Agriculture Advisor</div>", unsafe_allow_html=True)
st.markdown("<p class='caption-text'>A comprehensive tool for modern Indian farming. Get recommendations on crop choice, profitability, and sustainability.</p>", unsafe_allow_html=True)

# --- SIDEBAR FOR INPUTS ---
with st.sidebar:
    st.image("https://i.imgur.com/ag3rpA1.png", use_container_width=True)
    st.header("📍 Farm Inputs")

    with st.expander("🌍 Environmental Conditions", expanded=True):
        ph = st.slider("Soil pH", 5.0, 8.0, 6.8, 0.1)
        moisture = st.slider("Soil Moisture (%)", 10.0, 50.0, 32.0, 0.5)
        temp = st.slider("Avg. Temperature (°C)", 15.0, 35.0, 26.0, 0.5)
        rain = st.slider("Seasonal Rainfall (mm)", 50.0, 300.0, 175.0, 5.0)
        loc = st.text_input("Location (e.g., Northern India)", "Northern India")

    with st.expander("💰 Economic Factors (in INR)"):
        # Sliders now accept input in Rupees
        price = st.slider("Est. Market Price (₹/ton)", 15000, 40000, 20000, 500)
        cost = st.slider("Est. Cultivation Cost (₹/ha)", 50000, 150000, 90000, 1000)

    st.info("Adjust the sliders to see recommendations update in real-time.")
    if not loc.strip(): st.warning("Please provide a location.")

# --- MAIN CONTENT AREA ---
if not loc.strip():
    st.error("Please enter a location in the sidebar to get recommendations.")
else:
    inputs = {'ph': ph, 'moisture': moisture, 'temp': temp, 'rain': rain, 'loc': loc, 'price': price, 'cost': cost}
    with st.spinner('Analyzing conditions and running models...'):
        recommendations = generate_recommendations(inputs, model, le)
    
    top_rec = recommendations[0]

    tab1, tab2, tab3 = st.tabs(["🏆 Top Recommendation Dashboard", "📊 Comparative Analysis", "🧠 About the Model"])

    with tab1:
        st.markdown(f"<div class='subheader'>Top Recommendation: {top_rec['Crop']}</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(top_rec['Image'], caption=f"Recommended Crop: {top_rec['Crop']}", use_container_width=True)
        with col2:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                # Displaying profit in Rupees
                create_metric_card("Estimated Profit", f"₹{top_rec['Profit']:,.0f}", "/ ha", "💰")
                create_metric_card("Predicted Yield", f"{top_rec['Yield']:.2f}", "tons / ha", "🌾")
            with sub_col2:
                create_metric_card("Sustainability Score", f"{top_rec['Sustainability']:.0f}%", "Optimal Match", "💚")
                create_metric_card("Planting Season", top_rec['Planting Time'], f"in {loc}", "🗓️")

        st.markdown("<div class='subheader'>Actionable Environmental Advice</div>", unsafe_allow_html=True)
        for advice in top_rec['Advice']:
            st.markdown(advice)

    with tab2:
        st.markdown("<div class='subheader'>Yield, Profit & Sustainability Comparison</div>", unsafe_allow_html=True)
        
        df_recs = pd.DataFrame(recommendations)
        base = alt.Chart(df_recs).encode(x=alt.X('Crop:N', sort='-y', title='Crop Type'))
        
        bar_yield = base.mark_bar().encode(
            y=alt.Y('Yield:Q', title='Predicted Yield (tons/ha)'),
            color=alt.value('#138808'), # Green
            tooltip=['Crop', alt.Tooltip('Yield:Q', format='.2f')]
        )
        # Updated axis title and tooltip format for Rupees
        line_profit = base.mark_line(color='#FF9933', point=True, size=3).encode(
            y=alt.Y('Profit:Q', title='Estimated Profit (₹/ha)', axis=alt.Axis(orient='right')),
            tooltip=['Crop', alt.Tooltip('Profit:Q', format='₹,.0f')]
        )
        circle_sustainability = base.mark_circle(size=150, opacity=0.8, color='#000080').encode( # Navy Blue
            y=alt.Y('Yield:Q'),
            size=alt.Size('Sustainability:Q', title='Sustainability Score (%)', scale=alt.Scale(range=[50, 500])),
            tooltip=['Crop', alt.Tooltip('Sustainability:Q', format='.1f')]
        )
        chart = alt.layer(bar_yield, circle_sustainability, line_profit).resolve_scale(
            y='independent'
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)

    with tab3:
        st.markdown("<div class='subheader'>Model & Data Insights</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Model R² Score", f"{r2:.2f}")
        col2.metric("Model Mean Absolute Error", f"{mae:.2f} tons/ha")
        st.markdown("**Feature Importance:**")
        importances = pd.Series(model.feature_importances_, index=['Soil pH', 'Soil Moisture', 'Temperature', 'Rainfall', 'Crop Type']).sort_values(ascending=False)
        st.bar_chart(importances)
        st.warning("**Disclaimer:** This tool is for advisory purposes only. Market prices, costs, and environmental conditions can vary. Always consult with local agricultural experts before making financial decisions.")
