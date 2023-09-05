# -------- IMPORTS -------------
import json
import sys

from loguru import logger
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from main import (
    get_img_from_bytes,
    model_detect,
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


# Redirect to Swagger docs
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


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
    predict = model_detect(input_image)

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
        A success JSON reponse displaying image object detection successful.
    """

    # Get image from bytes
    input_image = get_img_from_bytes(file)

    # Model predict
    predict = model_detect(input_image)

    return {"Success!": "Image object detection successful."}
