
import streamlit as st
import requests

st.set_page_config(
    page_title="Crop Prediction",
    layout="wide"
)

API_URL = "http://api:8000/predict"

st.title("Crop Prediction")
st.write("Predict agricultural production using our trained Machine Learning model.")

# Input fields
state = st.selectbox(
    "State",
    [
        "Punjab",
        "Karnataka",
        "Tamil Nadu",
        "Kerala",
        "Maharashtra"
    ]
)

district = st.text_input(
    "District",
    value="LUDHIANA"
)

crop_year = st.number_input(
    "Crop Year",
    min_value=1900,
    max_value=2100,
    value=2020,
    step=1
)

season = st.selectbox(
    "Season",
    [
        "Kharif",
        "Rabi",
        "Whole Year"
    ]
)

crop = st.text_input(
    "Crop",
    value="Wheat"
)

area = st.number_input(
    "Area",
    min_value=0.01,
    value=100.0,
    step=1.0
)

# Prediction button
if st.button("Predict", type="primary"):

    if not district.strip() or not crop.strip():
        st.error("Please enter both district and crop.")

    else:
        payload = {
            "State_Name": state,
            "District_Name": district.strip(),
            "Crop_Year": int(crop_year),
            "Season": season,
            "Crop": crop.strip(),
            "Area": float(area)
        }

        try:
            with st.spinner("Generating prediction..."):

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

            st.success("Crop production prediction generated successfully!")

            st.metric(
                "Predicted Production",
                f"{result['predicted_production']:,.2f}"
            )

            st.caption(
                "Unit: Same measurement unit as the dataset's Production column."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Please make sure the API server is running on port 8000."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"Prediction request failed: {error}")
