import streamlit as st
import requests
from PIL import Image
import os


def streamlit_frontend():
    # Streamlit frontend for Object Detection.
    st.title("Object Detection with Streamlit")

    # Upload a file
    image_file = st.file_uploader("Upload an image.")

    # Predict object detections.
    if image_file:
        # Get the file extension
        file_extension = image_file.name.split(".")[-1]

        # Check the file extension
        if file_extension in ["jpg", "png", "gif", "tiff", "jpeg"]:
            response = requests.post(
                "http://localhost:8008/img_obj_detection_to_json",
                files={"file": image_file},
            )
            result = response.json()
            # Display object detection.
            st.write("Detected Objects:")
            st.write(result["detect_objects"])

            results = requests.post(
                   "http://localhost:8008/img_obj_detection_to_img",
                    files={"file": image_file},
            )

            st.subheader("Annotated Image")
            annotated_image_path = os.path.join(os.getcwd(),"annotated_image.jpg")
            annotated_image = Image.open(annotated_image_path)
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        else:
            st.error("Unsupported file format: {}".format(file_extension))


if __name__ == "__main__":
    streamlit_frontend()
