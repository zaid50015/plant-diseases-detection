import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv() 
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini client
client = genai.Client(api_key=gemini_api_key)

# Available Indian languages for translation
languages = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Tamil": "ta",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "Urdu": "ur"
}

# Translation function
def translate_to_language(text, lang_name):
    try:
        lang_code = languages[lang_name]
        prompt = f"Translate the following plant disease name to {lang_name} ({lang_code}) in a short and simple way: {text}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text.strip().split('\n')[0]
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return "Translation failed"

# Predict function
def model_prediction(test_image):
    try:
       model = tf.keras.models.load_model("plant-diseases-model.keras")
       image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
       input_arr = tf.keras.preprocessing.image.img_to_array(image)
       input_arr = np.array([input_arr]) #convert single image to batch
       predictions = model.predict(input_arr)
       return np.argmax(predictions) #return index of max element
       
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Remedies function
def get_remedies(disease_name):
    try:
        prompt = f"List some simple and practical remedies in bullet points for the plant disease: {disease_name}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
    except Exception as e:
        st.error(f"Error fetching remedies: {str(e)}")
        return "Could not retrieve remedies."

# UI starts here
st.header("🌿 Plant Disease Recognition & Remedies")

# Language selection dropdown
selected_language = st.selectbox("🌐 Select language for translation", list(languages.keys()))

# Image upload
test_image = st.file_uploader("📷 Upload a plant leaf image:", type=['jpg', 'jpeg', 'png'])

# Show image preview
if test_image is not None:
    st.image(test_image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Prediction button
    if st.button("🔍 Predict"):
        st.snow()
        st.write("⏳ Running model...")

        result_index = model_prediction(test_image)

        if result_index is not None:
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            predicted_disease = class_name[result_index]
            translated_name = translate_to_language(predicted_disease, selected_language)

            # Store in session state
            st.session_state.predicted_disease = predicted_disease
            st.session_state.translated_name = translated_name
            

            st.success(f"✅ Model's Prediction: {predicted_disease}")
            st.info(f"🌐 Translated to {selected_language}: {translated_name}")

# Show results and remedies button if prediction exists
if 'predicted_disease' in st.session_state:
    if "healthy" not in st.session_state.predicted_disease.lower():
        if st.button("🧪 Show Remedies"):
            with st.spinner("Fetching remedies..."):
                remedies = get_remedies(st.session_state.predicted_disease.replace("_", " "))
                st.subheader("✅ Remedies:")
                st.markdown(remedies)
