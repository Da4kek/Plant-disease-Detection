import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.metrics import classification_report

model = tf.keras.models.load_model('models/cnn_fuzzy_model.h5')

class_labels = os.listdir(
    "dataset/Citrus Leaf Disease Image/train")

def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant to predict the disease.")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False)

        if st.button('Predict'):
            image = image.resize((150, 150))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)[0]

            predicted_class_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_index]

            st.write(f"Predicted Class: {predicted_class}")
            st.write("Confidence:")
            for i, label in enumerate(class_labels):
                st.write(f"{label}: {round(prediction[i] * 100, 2)}%")
            
            plt.bar(class_labels, prediction * 100)
            st.set_option("deprecation.showPyplotGlobalUse", False)
            plt.xlabel('Class Labels')
            plt.ylabel('Confidence (%)')
            plt.title('Confidence Scores')
            st.pyplot()
            


if __name__ == '__main__':
    main()
