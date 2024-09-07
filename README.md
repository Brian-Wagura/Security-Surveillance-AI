#  SECURITY SURVEILLANCE AI INTERNSHIP PROJECT

# IMAGE DETECTION 

### Description

This project is a full-stack application that allows users to upload an image, detect objects within it using the YOLOv8 object detection model, and display the results to the user. The backend is built using FastAPI for handling image processing and serving the YOLOv8 model, while the frontend is developed with Streamlit for a simple and user-friendly interface.

### Features

* Image Upload: Users can upload images through the Streamlit interface.
* Object Detection: YOLOv8 model is used to detect objects in the uploaded images.
* Results Display: Detected objects and their confidence scores are displayed on the Streamlit app, along with the processed image showing bounding boxes.
* FastAPI Backend: FastAPI serves the object detection model and processes the image.
* Logging: The application logs all requests and predictions for monitoring and debugging.

### Architecture

* Frontend: Streamlit handles the user interface, allowing users to upload images and view the detection results.
* Backend: FastAPI processes the image and runs the YOLOv8 model to detect objects. The model inference is handled in the backend, and the results are returned to the frontend.
* Model: YOLOv8 (You Only Look Once, Version 8) is used for real-time object detection.

## Installation

### Prerequisites
* Python 3.10

### Step by step guide

1. Clone the repository.
```
git clone https://github.com/Brian-Wagura/Security-Surveillance-AI.git
cd Security-Surveillance-AI
```
2. Create and activate virtual environment.
```
python -m venv virtualenvironment
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install dependencies.
```
pip install -r requirements.txt
```
4. Run FASTAPI backend.
```
uvicorn api:app --reload
```
5. Run Streamlit Frontend: Open a new terminal window and execute:
```
streamlit run app.py
```

## Usage

##### Start the Application:

* First, start the FastAPI server by running the uvicorn command mentioned in the installation section.
* Then, start the Streamlit app using the streamlit run command.

##### Upload an Image:

* Once the Streamlit app is running, navigate to the URL provided by Streamlit (usually http://localhost:8501).
* Upload an image using the file uploader in the app.

##### View Results:

* After uploading the image, the app will send the image to the FastAPI backend, where YOLOv8 detects objects.
* The detected objects, along with their confidence scores, will be displayed on the app.
* The processed image with bounding boxes around detected objects will also be displayed.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.