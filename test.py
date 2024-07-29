
import tensorflow as tf
import numpy as np
# from PIL import Image
import sys

# print(tf.__version__)
# version = sys.version
# print(f"Current Python version: {version}")


def load_model():
    try:
        model = tf.keras.models.load_model('plant-diseases-model.keras')
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

model = load_model()
if model!=None:
    print("Model loaded successfully")