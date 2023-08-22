# -------- IMPORTS -------------
import json
import pandas as pd

from PIL import Image
from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from ultralytics import YOLO

from main import get_img_from_bytes

# --------- LOGGER ---------------


# ---------FASTAPI SETUP ---------
app = FastAPI(
    title="Object Detection with FastAPI",
    description="Obtain object value out of image \
        and return image and JSON response.",
)

origins = ["http://localhost", "http://localhost:8008", "*"]


@app.on_event("startup")
def save_openapi_json():
    """
    This function is used to save the OpenAPI documentation
    data of the FastAPI application to a JSON file.
    The purpose of saving the OpenAPI documentation data is to have
    a permanent and offline record of the API specification,
    which can be used for documentation purposes or
    to generate client libraries
    """

    openapi_data = app.openapi()

    with open("openapi.json", "w") as doc_file:
        json.dump(openapi_data, doc_file)


# Redirect to Swagger docs
app.get("/", include_in_schema=False)


async def redirect():
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def perform_healthcheck():
    """
    Sends a GET request to the route & hopes to get a "200"
    response code.  It acts as a last line of defense in
    case something goes south. Additionally, it also
    returns a JSON response in the form of:
    {
        'healthcheck': 'Everything OK!'
    }
    """
    return {"healthcheck": "Everything OK!"}


# --------- SUPPORT FUNCTION ---------------


def get_model_predict(
    model: YOLO,
    input_image: Image,
    save: bool = False,
    image_size: int = 1248,
    conf: float = 0.5,
    augment: bool = False,
) -> pd.DataFrame:
    
    """
    Get the predictions of a model on an input image.

    Args:
        model(YOLO) - The trained YOLO model.
        input_image(Image) - The image on which the model will make predictions.

    """
    pass


# --------- MAIN FUNCTION ------------------


@app.post("/img_obj_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object detection from an Image.

    Args: file(bytes) - The img file in bytes format.
    Returns: dict - JSON format containing the object detections.
    """

    # Result dictionary with None values
    result = {"detect_objects": None}

    # Convert the image file to an Image object
    input_image = get_img_from_bytes(file)

    # Predict from model
