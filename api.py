# -------- IMPORTS -------------
import os
import json
import sys

from loguru import logger
from fastapi import FastAPI, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from main import (
    get_img_from_bytes,
    detect_sample_model,
    add_bboxs_on_img,
)

# --------- LOGGER ---------------

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")


# ---------FASTAPI SETUP ---------
app = FastAPI(
    title="Object Detection with FastAPI",
    description="Obtain object value out of image \
        and return JSON response and image result.",
)

origins = ["http://localhost", "http://localhost:8008", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def save_openapi_json():
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
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def perform_healthcheck():
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


# --------- MAIN ROUTES ------------------


@app.post("/img_obj_detection_to_json")
async def img_object_detection_to_json(file: bytes = File(...)):
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
    predict = detect_sample_model(input_image)

    # Select detect obj return info
    detect_res = predict[["name", "confidence"]]
    objects = detect_res["name"].values

    result["detect_objects_names"] = ", ".join(objects)
    result["detect_objects"] = json.loads(detect_res.to_json(orient="records"))

    # Logs and return
    logger.info("results: {}", result)
    return result


@app.post("/img_obj_detection_to_img")
async def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object detection from an image, plot bbox on image

    Args:
        file (bytes) - The image file in bytes format
    Returns:
        Image - Image in bytes with bbox annotations
    """

    # Get image from bytes
    input_image = get_img_from_bytes(file)

    # Model predict
    predict = detect_sample_model(input_image)

    # Add bbox on image
    final_image = add_bboxs_on_img(image=input_image, predict=predict)

    return final_image
