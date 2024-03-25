import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('models/cnn_model.h5')

# Define the class labels
# Replace 'class3' and 'class4' with your actual class labels
class_labels = ['black-spot', 'citrus canker', 'fresh', 'grenning']


def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant to predict the disease.")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            # Preprocess the image
            image = image.resize((150, 150))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Make prediction
            prediction = model.predict(image)[0]

            # Determine the predicted class label
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_index]

            # Print the predicted class label and confidence
            st.write(f"Predicted Class: {predicted_class}")
            st.write("Confidence:")
            for i, label in enumerate(class_labels):
                st.write(f"{label}: {round(prediction[i] * 100, 2)}%")


if __name__ == '__main__':
    main()
