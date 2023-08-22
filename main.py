# -------------- IMPORTS --------------
import io
import pandas as pd

from PIL import Image

# -------------------------------------


# -------------------------------------


def get_img_from_bytes(binary_img: bytes) -> Image:
    """
    Converts image from bytes to PIL RGB format.

    Args: binary_image(bytes) - Binary representation of the image.
    Returns: PIL.Image - The image in PIL RGB format
    """

    input_image = Image.open(io.BytesIO(binary_img)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to bytes

    Args: image(Image) - A PIL image instance
    Returns: bytes - BytesIO object that contains the image in JPEG format
            with quality 85
    """

    return_image = io.BytesIO()

    # Save the image in JPEG format with quality 85
    image.save(return_image, format="JPEG", quality=85)

    # Set the pointer to the beginning of the file
    return_image.seek(0)

    return return_image


def transform_predict_to_df(results: list, labels_dict: dict) -> pd.DataFrame:
    """
    Transform prediction from yolov8 (torch.Tensor) to pandas DataFrame

    Args: results(list) - A list containing the predict output from yolov8 in the
                        form of a torch.Tensor
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
):
    pass


#  --------------- MODELS -----------------


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model
    Based on Yolov8

    Args: input_image (Image) - the input image
    Returns: pd.DataFrame - DataFrame containing the object location.
    """

    predict = get_model_predict()
