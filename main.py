import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('models/cnn_model.h5')

# Create the web application


def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant to predict the disease.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Check if the predict button is clicked
        if st.button('Predict'):
            # Preprocess the image
            image = image.resize((150, 150))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Make predictions
            prediction = model.predict(image)[0][0]

            # Display the prediction
            if prediction >= 0.5:
                st.write("Prediction: Citrus Canker")
                st.write(f"Accuracy: {round(prediction * 100, 2)}%")
            else:
                st.write("Prediction: Black Spot")
                st.write(f"Accuracy: {round((1 - prediction) * 100, 2)}%")


# Run the web application
if __name__ == '__main__':
    main()
