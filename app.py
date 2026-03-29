import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Antibiotic Resistance Prediction System",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .recommendation-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        background-color: #e9f7ef;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to load models
@st.cache_resource
def load_models_and_columns():
    models = {}
    antibiotics = ['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']
    
    # Load models
    for anti in antibiotics:
        model_path = f"{anti}_model.pkl"
        if os.path.exists(model_path):
            try:
                models[anti] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model for {anti}: {e}")
        else:
            st.warning(f"Model file {model_path} not found.")
            
    # Load columns
    columns = None
    if os.path.exists('columns.pkl'):
        try:
            columns = joblib.load('columns.pkl')
        except Exception as e:
            st.error(f"Error loading columns.pkl: {e}")
    
    # Fallback/Default columns based on project context if columns.pkl is missing
    if columns is None:
        # Based on Dataset.xlsx and training logic (get_dummies with drop_first=True)
        # Locations: EDE-C, EDE-S, EDE-T, IFE-C, IFE-S, IFE-T, IWO-C, IWO-S, IWO-T, OSU-C, OSU-S, OSU-T
        # If drop_first=True, EDE-C is dropped.
        locations = ['EDE-S', 'EDE-T', 'IFE-C', 'IFE-S', 'IFE-T', 'IWO-C', 'IWO-S', 'IWO-T', 'OSU-C', 'OSU-S', 'OSU-T']
        location_cols = [f"Location_{loc}" for loc in locations]
        all_antibiotics = ['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']
        columns = all_antibiotics + location_cols
        
    return models, columns

# Load resources
models, training_columns = load_models_and_columns()

# UI Header
st.title("🧬 Antibiotic Resistance Prediction System")
st.markdown("""
    This is a **clinical decision support tool** driven by machine learning to predict the resistance of various antibiotics.
    Please input the location and the available clinical parameters below to get predictions and recommendations.
""")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

selected_location = st.sidebar.selectbox(
    "Location",
    ["IFE-T", "IFE-C", "IFE-S", "OSU-T", "OSU-C", "OSU-S", "IWO-T", "IWO-C", "IWO-S", "EDE-T", "EDE-C", "EDE-S"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Antibiotic Measurements (Continuous)")

val_ceftazidime = st.sidebar.number_input("CEFTAZIDIME", min_value=0.0, value=20.0, step=0.1)
val_gentamicin = st.sidebar.number_input("GENTAMICIN", min_value=0.0, value=20.0, step=0.1)
val_augmentin = st.sidebar.number_input("AUGMENTIN", min_value=0.0, value=20.0, step=0.1)
val_ciprofloxacin = st.sidebar.number_input("CIPROFLOXACIN", min_value=0.0, value=25.0, step=0.1)

# Main Page Layout
col1, col2 = st.columns([1, 1])

if st.sidebar.button("Predict Resistance"):
    with st.spinner("Analyzing patterns and generating predictions..."):
        # 1. Preprocessing Input
        # Create initial dict with inputs we have
        input_data = {
            'CEFTAZIDIME': val_ceftazidime,
            'GENTAMICIN': val_gentamicin,
            'AUGMENTIN': val_augmentin,
            'CIPROFLOXACIN': val_ciprofloxacin,
            'Location': selected_location
        }
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # One-hot encoding for Location
        df_encoded = pd.get_dummies(df_input, columns=['Location'])
        
        # Ensure all columns from training are present
        for col in training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # 2. Prediction Logic
        results = []
        
        for name, model in models.items():
            # Prepare feature set for this specific model (drop the target if it exists)
            # However, the user requirement says "reorder columns exactly as in columns.pkl"
            # and "ensure missing columns are added with value 0".
            # Training code used X = df_encoded.drop(columns=[target]).
            
            # Reorder to match training columns
            # But the model expects feature_names_in_ from training.
            if hasattr(model, 'feature_names_in_'):
                features_to_use = model.feature_names_in_
            else:
                # If feature names not available, we use the global list minus the current target
                features_to_use = [c for c in training_columns if c != name]
            
            # Construct feature vector
            X_pred = df_encoded[features_to_use]
            
            # Predict
            pred = model.predict(X_pred)[0]
            
            # Probability (Confidence)
            prob = 0.0
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_pred)[0]
                prob = probs[pred] # Probability of predicted class
            
            results.append({
                "Antibiotic": name,
                "Prediction": pred,
                "Confidence": prob
            })
            
        # 3. Display Results
        with col1:
            st.subheader("📊 Prediction Results")
            
            for res in results:
                status = "Sensitive" if res['Prediction'] == 1 else "Resistant"
                icon = "✅" if res['Prediction'] == 1 else "❌"
                color = "green" if res['Prediction'] == 1 else "red"
                
                st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style="margin:0;">{res['Antibiotic']}</h4>
                        <p style="font-size: 20px; color: {color}; font-weight: bold;">
                            {icon} {status} <span style="font-size: 14px; color: grey;">(Confidence: {res['Confidence']:.2%})</span>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("💡 Recommendation")
            
            # Filtering sensitive antibiotics
            sensitive_options = [r for r in results if r['Prediction'] == 1]
            
            if sensitive_options:
                # Recommend the best antibiotic (highest confidence)
                best_option = max(sensitive_options, key=lambda x: x['Confidence'])
                
                st.success(f"**Recommended Antibiotic:** {best_option['Antibiotic']}")
                st.markdown(f"""
                    <div class="recommendation-box">
                        Based on the predictive model, <b>{best_option['Antibiotic']}</b> shows the highest likelihood 
                        of being effective for this specific case with <b>{best_option['Confidence']:.2%}</b> confidence.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("No safe antibiotic predicted — further testing required")
                st.warning("All tested antibiotics are predicted to be Resistant (❌). Please consult clinical guidelines for alternative treatments.")

            # 4. Visualization
            st.markdown("---")
            st.subheader("📉 Confidence Comparison")
            
            prob_df = pd.DataFrame(results)
            # Use probability of 'Sensitive' class for the chart to show potential effectiveness
            # Since pred=1 is sensitive, if pred=1, prob = prob_1. If pred=0, prob = prob_0 (1 - prob_1).
            # We want to show prob_1 (likelihood of sensitivity) for all.
            
            chart_data = []
            for res in results:
                # We need prob_1
                name = res['Antibiotic']
                model = models.get(name)
                if model and hasattr(model, "predict_proba"):
                    if hasattr(model, 'feature_names_in_'):
                        features_to_use = model.feature_names_in_
                    else:
                        features_to_use = [c for c in training_columns if c != name]
                    p1 = model.predict_proba(df_encoded[features_to_use])[0][1]
                    chart_data.append({"Antibiotic": name, "Sensitivity Probability": p1})
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                st.bar_chart(chart_df.set_index("Antibiotic"))
            else:
                st.info("Probability data not available for visualization.")

else:
    st.info("👈 Fill in the parameters in the sidebar and click **Predict Resistance** to start.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: grey;">
        <p>© 2026 Healthcare AI - Antibiotic Resistance Prediction System</p>
    </div>
""", unsafe_allow_html=True)
