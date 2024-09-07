import os
import requests
import streamlit as st

base_directory = "runs/detect/"


def create_uploads_folder():
    """
    Creates an "uploads" folder if it does not exist
    """
    if not os.path.exists("uploads"):
        os.makedirs("uploads")


def save_uploaded_image(image_file):
    """
    Save the uploaded image to the "uploads" folder
    """
    file_path = os.path.join("uploads", image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.read())
    return file_path


def display_results_json(image_path):
    try:
        response = requests.post(
            "http://localhost:8000/img_obj_detection",
            files={"file": open(image_path, "rb")},
        )
        result = response.json()
        # Display object detection.
        st.write("Detected Objects:")
        st.write(result["detect_objects"])
    except requests.exceptions.HTTPError as e:
        st.error(e)


def find_latest_prediction_folder(base_directory):
    # List all subdirectories in the base directory
    subdirectories = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]

    # Filter subdirectories that start with predict
    prediction_folders = [d for d in subdirectories if d.startswith("predict")]

    # Sort to find the most recent
    prediction_folders.sort(reverse=True)

    if prediction_folders:
        return os.path.join(base_directory, prediction_folders[0])
    else:
        return None


def display_results_image():
    try:
        st.subheader("Predicted Image")
        latest_prediction_folder = find_latest_prediction_folder(base_directory)

        if latest_prediction_folder:
            res_img_path = os.path.join(latest_prediction_folder, "image0.jpg")
            st.image(res_img_path, caption="Predicted Image", use_column_width=True)
        else:
            st.warning("No prediction folders found.")
    except requests.exceptions.HTTPError as e:
        st.error(e)


def streamlit_frontend():
    # Streamlit frontend for Object Detection.
    st.title("Image Object Detection")

    # Create the uploads and predictions folder
    create_uploads_folder()

    # Upload a file
    image_file = st.file_uploader(
        "Upload an image.", type=["jpeg", "jpg", "png", "gif", "tiff"]
    )

    # Predict object detections.
    if image_file:
        # Save the uploaded image
        image_path = save_uploaded_image(image_file)

        # Display results in JSON format
        display_results_json(image_path)

        # Display the image with the predicted objects
        display_results_image()


if __name__ == "__main__":
    streamlit_frontend()
