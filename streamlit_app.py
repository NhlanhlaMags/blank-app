

import streamlit as st
import joblib
import pandas as pd
from io import StringIO
import os

# Configuration
st.set_page_config(
    page_title="Municipal Protest Risk Dashboard",
    page_icon="⚠️",
    layout="wide"
)

# Constants
PROVINCES = [
    'Eastern Cape', 'Free State', 'Gauteng', 'KwaZulu-Natal',
    'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape', 'Western Cape'
]

@st.cache_resource
def load_model():
    #Load the trained model pipeline
    model_path = "/content/protest_risk_model.pkl"
    return joblib.load(model_path)

model = load_model()

# Helper functions
def validate_inputs(df):
    #Ensure data quality before prediction
    # Convert province names
    if 'Province name' in df.columns:
        df['Province name'] = df['Province name'].str.strip().str.title()
        df['Province name'] = df['Province name'].apply(
            lambda x: x if x in PROVINCES else 'Unknown'
        )

    # Ensure numerical fields are valid
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].clip(lower=0)

# App layout
st.title("🇿🇦 South African Municipal Protest Risk Predictor")
st.markdown("
This tool assesses protest risk probability based on municipal characteristics.
")

tab1, tab2 = st.tabs(["Single Municipality", "Batch Processing"])

with tab1:
    with st.form("single_prediction"):
        st.subheader("Municipal Characteristics")

        cols = st.columns(3)
        with cols[0]:
            province = st.selectbox("Province", PROVINCES)
        with cols[1]:
            district = st.text_input("District Municipality")
        with cols[2]:
            local_muni = st.text_input("Local Municipality")

        st.subheader("Population Demographics")
        demo_cols = st.columns(5)
        with demo_cols[0]:
            total = st.number_input("Total Population", min_value=0, value=100000)
        with demo_cols[1]:
            black = st.number_input("Black African", min_value=0, value=80000)
        with demo_cols[2]:
            coloured = st.number_input("Coloured", min_value=0, value=10000)
        with demo_cols[3]:
            indian = st.number_input("Indian/Asian", min_value=0, value=5000)
        with demo_cols[4]:
            white = st.number_input("White", min_value=0, value=5000)

        st.subheader("Living Conditions")
        hs_cols = st.columns([2,1,1,1,1])
        with hs_cols[0]:
            informal = st.number_input("Informal Dwellings", min_value=0, value=5000)
        with hs_cols[1]:
            piped_water = st.number_input("Piped Water Access", min_value=0, value=70000)
        with hs_cols[2]:
            no_water = st.number_input("No Water Access", min_value=0, value=10000)
        with hs_cols[3]:
            pit_toilet = st.number_input("Pit Toilets", min_value=0, value=20000)
        with hs_cols[4]:
            bucket_toilet = st.number_input("Bucket Toilets", min_value=0, value=1000)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = {
            'Province name': province,
            'District municipality name': district,
            'District/Local municipality name': local_muni,
            'Local municipality code': 0,  # Placeholder
            'ID': 0,  # Placeholder
            'Total': total,
            'Black African': black,
            'Coloured': coloured,
            'Indian/Asian': indian,
            'White': white,
            'Informal Dwelling': informal,
            'Piped (tap) water on community stand': piped_water,
            'No access to piped (tap) water': no_water,
            'Pit toilet': pit_toilet,
            'Bucket toilet': bucket_toilet,
            'Communal refuse dump': 0,  # Defaults
            'Communal container/central collection point': 0,
            'Own refuse dump': 0,
            'Dump or leave rubbish anywhere (no rubbish disposal)': 0,
            'Gas': 0,
            'Paraffin': 0,
            'Candles': 0,
            'Paraffin_8': 0,
            'Wood': 0,
            'Coal': 0,
            'Animal dung': 0
        }

        input_df = pd.DataFrame([input_data])
        input_df = validate_inputs(input_df)

        try:
            # Access the model from the loaded dictionary for prediction
            risk_prob = model['model'].predict_proba(input_df)[0][1] * 100

            st.success(f"Predicted Protest Risk: {risk_prob:.1f}%")

            # Visual indicators
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Level",
                         "High" if risk_prob > 70 else
                         "Medium" if risk_prob > 40 else "Low")
            with col2:
                st.progress(int(risk_prob))

            # Explanation
            st.plotly_chart(explain_prediction(input_df), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.subheader("Batch Prediction via CSV")

    # Access the feature names from the preprocessor within the model dictionary
    sample_template = pd.DataFrame(columns=model['preprocessor'].feature_names_in_)
    st.download_button(
        "Download CSV Template",
        sample_template.to_csv(index=False),
        "template.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = validate_inputs(df)

            # Access the feature names from the preprocessor within the model dictionary
            missing_cols = set(model['preprocessor'].feature_names_in_) - set(df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {list(missing_cols)}")
            else:
                # Access the model from the loaded dictionary for prediction
                df = df[model['preprocessor'].feature_names_in_]
                predictions = model['model'].predict_proba(df)[:, 1] * 100
                df['Protest Risk (%)'] = predictions.round(1)

                st.success(f"Processed {len(df)} records")

                # Show top 5 risk areas
                st.dataframe(
                    df.sort_values('Protest Risk (%)', ascending=False)
                    .head(5)
                    .style.background_gradient(
                        cmap='Reds',
                        subset=['Protest Risk (%)']
                    )
                )

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.caption("
Protest Risk Prediction Model v1.0
Data sources: Census, Municipal Reports
")
