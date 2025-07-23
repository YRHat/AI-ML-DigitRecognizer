import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf


# Load your trained model
model = tf.keras.models.load_model("my_cnn_mnist_model.h5")

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) below and let the model predict it!")

# Canvas settings
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",

)


if canvas_result.image_data is not None:
    img = canvas_result.image_data


    # Convert to grayscale (0–255)
    img = Image.fromarray((255 - img[:, :, 0]).astype('uint8')).resize((28, 28))
    
     # ✅ Reshape correctly for CNN input
    img_array_check = np.array(img)

    #for no initial display of predicted value

    if np.all(img_array_check == 0):  # All black image → nothing drawn
        st.warning("✍️ Please draw a digit to get a prediction.")
    else:
        img_array = img_array_check.reshape(1, 28, 28, 1).astype("float32")
        img_array /= 255.0
        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.subheader(f"✏️ Predicted Digit: **{predicted_class}**")










