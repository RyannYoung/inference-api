from flask import Flask, request, send_file
from typing import Literal
from handling import api_error, format_response
from lib import (
    classify,
    classify_group,
    get_image,
    execute_with_threadpool,
)
from tesseract import evaluate_image
import os
from PIL.Image import Image
from models.model_controller import ModelController

dir_path = os.path.dirname(os.path.realpath(__file__))

# Classification Models
Classifcation = Literal["classify", "ocr"]
ClassifyModel = Literal["ResNet500"]

# Initialise the flask app
app = Flask(__name__)


# Global reference to the model controller
model_controller = ModelController()
model_controller.load_models()


@app.route("/")
def root():
    """Root dir"""
    return send_file(f"{dir_path}/../assets/index.html")


@app.route("/models")
def models():
    """Return information about the current models loaded"""
    return model_controller.display_models()


@app.route("/enrich", methods=["GET", "POST"])
def enrich():
    """Classify a single entity through a provided mode and type

    Example:
        "http://app/classify?src=https://image/image.png?mode=classify?model?=ResNet

    Returns:
        A generic enrichment endpoint which requires
        a source and type classifier.

        Note: Additional query parameters may be required
        for more complex models.
    """

    res = None  # The response message
    format = None  # The format message

    # GET REQUEST
    if request.method == "GET":
        """Extract params the query keys"""
        # Source Param & Guard
        src = request.args.get("src")
        if src is None:
            return api_error("url query parameter was not provided")
        else:
            src.strip()

        img = get_image(src)
        if img is None:
            return api_error("could not generate image from src query paramter")

        # Mode Param & Guard
        enrich_mode = request.args.get("mode", default="classify").strip().lower()

        # Model
        model = request.args.get("model", default="ResNet").strip()
        model = model_controller.find_model(model)  # Find the model from the controller

        if model is None:
            return api_error("model query parameter did not match any loaded models")

        # Return format
        format = request.args.get("format", default="img").strip()

        # Run the specific mode based on the query param
        match enrich_mode:
            case "classify":
                # Run Classification
                res = classify(img, model, format)
            case "ocr":
                # Run OCR
                res = evaluate_image(img)
            case _:
                return api_error("'mode query parameter was invalid")

    # POST REQUEST
    if request.method == "POST":
        """Extract params from the POST body"""

        json = request.json
        if json is None:
            return api_error("invalid JSON in POST body")

        src = json.get("src")
        if src is None:
            return api_error("src was not specified as a POST body parameter")

        model = json.get("model", "ResNet")
        model = model_controller.find_model(model)  # Find the model from the controller
        if model is None:
            return api_error(
                f"'model' query parameter did not match any loaded models. Available models {model_controller.display_available_models()}"
            )

        format = json.get("format", "default")
        enrich_mode = json.get("mode", "classify")

        # Create images
        images = [get_image(url) for url in src]

        # Filter out the failed images
        images_filtered: list[Image] = list(filter(lambda x: x is not None, images))  # type: ignore

        if images_filtered is None:
            return api_error("No images available to process post filter")

        match enrich_mode:
            case "classify":
                res = classify_group(images_filtered, model, format)
            case "ocr":
                return api_error("OCR is currently unimplemented!")
            case _:
                return api_error("'mode query parameter was invalid")

    return format_response(res, format)
