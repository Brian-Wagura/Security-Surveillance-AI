# -------------- IMPORTS --------------
import io
import pandas as pd
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image

# ------------ YOLO Model -----------------

# Initialize the model
model_sample = YOLO("./models/yolov8m.pt")

# ------------ MAIN FUNCTIONS --------------------


def get_img_from_bytes(binary_img: bytes) -> Image:
    """
    Converts image from bytes to PIL RGB format.

    Args: binary_image(bytes) - Binary representation of the image.
    Returns: PIL.Image - The image in PIL RGB format
    """

    input_image = Image.open(io.BytesIO(binary_img)).convert("RGB")
    return input_image


def transform_predict_to_df(results: list, labels_dict: dict) -> pd.DataFrame:
    """
    Transform prediction from yolov8 (torch.Tensor) to pandas DataFrame

    Args: results(list) - A list containing the predict output from yolov8
          in the form of a torch.Tensor
          labels_dict(dict) -  A dictionary containing the labels names, where the
          keys are the class ids and the values are the label names.

    Returns: predict_bbox (pd.DataFrame) - A DataFrame containing the bounding box
                    coordinates, confidence scores and class labels.

    """

    # Transform Tensor to numpy array
    predict_bbox = pd.DataFrame(
        results[0].to("cpu").numpy().boxes.xyxy,
        columns=["xmin", "ymin", "xmax", "ymax"],
    )

    # Add the confidence of the prediction to the DataFrame
    predict_bbox["confidence"] = results[0].to("cpu").numpy().boxes.conf

    # Add the class of the prediction to the DataFrame
    predict_bbox["class"] = (results[0].to("cpu").numpy().boxes.cls).astype(int)

    # Replace the class number with the class name from the labels_dict
    predict_bbox["name"] = predict_bbox["class"].replace(labels_dict)

    return predict_bbox


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
        save (bool,optional) - Whether to save the image with the predictions. Defaults to false.
        image_size (int,optional) - The size of the image the model will receive. Defaults to 1248.
        conf (float, optional) - The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional) - Whether to apply data augmentation on the input image. Defaults to false.

    Returns:
        pd.DataFrame: A dataframe containing the predictions.

    """

    # Make predictions
    predictions = model.predict(
        imgsz=image_size,
        source=input_image,
        conf=conf,
        save=save,
        augment=augment,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
    )

    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)

    return predictions


#  ------------ BBOX FUNCTION ------------


def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    Add a bounding box on the Image.

    Args:
        image(Image) - input image
        predict(dataframe) - predict from model

    Returns:
        Image - image with bbox
    """

    # Annotator object
    annotator = Annotator(np.array(image))

    # Sort predict by xmin value
    predict = predict.sort_values(by=["xmin"], ascending=True)

    # Iterate over the rows of predict df
    for i, row in predict.iterrows():
        # Text to be displayed on the image.
        text = f"{row['name']}: {int(row['confidence']*100)}%"

        # Bounding box coordinates
        bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

        # Add bbox and text on the image
        annotator.box_label(bbox, text, color=colors(row["class"], True))

    # Convert the annotated image to PIL image
    return Image.fromarray(annotator.result())


#  --------------- MODEL -----------------


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model
    Based on Yolov8

    Args: input_image (Image) - the input image
    Returns: pd.DataFrame - DataFrame containing the object location.
    """

    predict = get_model_predict(
        model=model_sample,
        input_image=input_image,
        save=True,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict
