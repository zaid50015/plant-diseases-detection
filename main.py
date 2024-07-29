import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv() 

# Set up the Gemini API (make sure to keep your API key secret)
gemini_api_key=os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

def translate_to_hindi(text):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Translate the following plant disease name to Hindi: {text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return "Translation failed"

def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model("plant-diseases-model.keras")
        image = Image.open(test_image).convert('RGB')
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # return index of max element
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None



st.header("Plant Disease Recognition")
test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])

if test_image is not None:
    st.image(test_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        
        result_index = model_prediction(test_image)
        
        if result_index is not None:
            # Reading Labels
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
            hindi_translation = translate_to_hindi(predicted_disease)
            
            st.success(f"Model's prediction: {predicted_disease}")
            st.success(f"हिंदी में रोग का नाम: {hindi_translation}")
        else:
            st.warning("The model couldn't make a prediction.")
            
else:
    st.info("Please upload an image to begin.")