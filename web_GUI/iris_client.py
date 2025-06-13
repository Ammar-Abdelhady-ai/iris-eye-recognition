import streamlit as st
import requests

API_URL = "http://api:8000" 
#API_URL = "http://localhost:8600" 

st.set_page_config(page_title="Iris Eye Recognition", page_icon="👁️")
st.title("👁️ Iris Eye Recognition System")
st.markdown("""
This application allows you to:
- 🔍 Predict a user from their iris image
- 🧠 Train the model with new users
""")

st.sidebar.header("🔧 Actions")
mode = st.sidebar.radio("Select Mode", ["🔍 Predict", "🧠 Train"])

if mode == "🔍 Predict":
    st.header("🔍 Predict Identity from Eye Image")
    pred_image = st.file_uploader("Upload an eye image for prediction", type=["jpg", "jpeg", "png"], key="predict")

    if pred_image and st.button("Run Prediction"):
        files = {"file": pred_image.getvalue()}
        response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prediction Successful")
            st.markdown(f"**Predicted ID:** `{result.get('ID_Number', 'N/A')}`")
            st.markdown("### 👤 User Details")
            for key, value in result.items():
                st.write(f"**{key}**: {value}")
        else:
            st.error(response.json().get("error", "Prediction failed."))

elif mode == "🧠 Train":
    st.header("🧠 Train the Model with New Data")
    train_image = st.file_uploader("Upload an eye image for training", type=["jpg", "jpeg", "png"], key="train")

    with st.form("train_form"):
        st.subheader("User Info")
        name = st.text_input("Full Name")
        address = st.text_input("Address")
        id_number = st.text_input("ID Number")
        birth_date = st.date_input("Birth Date")
        reset_all = st.checkbox("🔁 Reset all data before training", value=False)
        submitted = st.form_submit_button("Add & Train")

    if train_image and submitted:
        if not all([name, address, id_number, birth_date]):
            st.warning("🚨 Please fill in all fields.")
        else:
            files = {"file": train_image.getvalue()}
            data = {
                "name": name,
                "address": address,
                "id_number": id_number,
                "birth_date": birth_date.isoformat(),
            }
            response = requests.post(f"{API_URL}/add_data_and_train", files=files, data=data)

            if response.status_code == 200:
                st.success(response.json().get("message", "Training completed!"))
            else:
                st.error(response.json().get("error", "Training failed."))
