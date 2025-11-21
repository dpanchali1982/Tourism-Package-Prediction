import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ==========================================
# 1. Load the Model from Hugging Face
# ==========================================
# NOTE: Replace '<---repo id---->' with your actual Hugging Face username
REPO_ID = "dpanchali/tourism-prediction-model"
MODEL_FILENAME = "best_tourism_pipeline.joblib"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# ==========================================
# 2. Streamlit UI Layout
# ==========================================
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package based on their profile and interaction details.
""")

if model:
    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=10000.0, value=20000.0, step=500.0)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        passport = st.selectbox("Has Passport?", ["No", "Yes"])
        own_car = st.selectbox("Owns Car?", ["No", "Yes"])

    with col2:
        st.subheader("Interaction Details")
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=0.0, value=10.0)
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, value=3.0)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        preferred_property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
        number_of_trips = st.number_input("Number of Trips Taken", min_value=0.0, value=2.0)
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)

    # ==========================================
    # 3. Data Processing & Prediction
    # ==========================================
    # Convert Yes/No to 1/0 for binary features if model expects it (assuming 1/0 based on typical CSVs)
    passport_val = 1 if passport == "Yes" else 0
    own_car_val = 1 if own_car == "Yes" else 0

    # Assemble input into DataFrame (Must match training feature names exactly)
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport_val,
        'PitchSatisfactionScore': pitch_satisfaction,
        'OwnCar': own_car_val,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation,
        'MonthlyIncome': monthly_income
    }])

    # Predict button
    if st.button("Predict Purchase Status"):
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]

            # Since it's a classifier, prediction is 0 or 1
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success("Customer is LIKELY to purchase the package.")
            else:
                st.warning("Customer is UNLIKELY to purchase the package.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
