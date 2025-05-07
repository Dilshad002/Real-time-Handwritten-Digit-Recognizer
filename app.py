import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("mnist_cnn_model.h5")

st.title("Handwritten Digit Recognizer (CNN)")

# Create canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert to image
        img = canvas_result.image_data.astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        inverted = cv2.bitwise_not(resized)
        normalized = inverted / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(input_img)
        st.write(f"### Prediction: {np.argmax(prediction)}")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit first.")
