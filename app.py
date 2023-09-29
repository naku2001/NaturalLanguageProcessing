import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tempfile
import pickle

# Load the pre-trained ResNet-50 model
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Define the prediction function
def predict_caption(photo):
    # ... your existing prediction logic here ...

# Streamlit app
def main():
    st.title("Image Caption")

    # Upload video file
    video_file = st.file_uploader("Upload Video ", type=["mp4"])
    predict_button = st.button("Predict")  # Add Predict button

    if predict_button and video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        # Convert video frames to images
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()

        # Process each frame and predict captions
        for i, frame in enumerate(frames):
            # Resize the frame to the input size of the ResNet-50 model
            frame = cv2.resize(frame, (224, 224))

            # Preprocess the image
            img = preprocess_input(frame)

            # Pass the image through the ResNet-50 model
            img_features = resnet_model.predict(np.expand_dims(img, axis=0))

            # Get the predicted caption
            caption = predict_caption(img_features)

            # Display the frame and the predicted caption
            st.image(frame, use_column_width=True)
            st.write(f"Caption for Frame {i+1}: {caption}")

# Run the Streamlit app
if __name__ == "__main__":
    main()

