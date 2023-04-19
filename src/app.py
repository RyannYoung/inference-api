from io import BytesIO
from flask import Flask, request, send_file, Response
from src.lib import (
    classify_image,
    classify_image_raw,
    get_image,
    grid_images,
    execute_with_threadpool,
)
from src.tesseract import evaluate_image

# Initialise the flask app
app = Flask(__name__)


@app.route("/image")
def image():
    """Provide AI label classifications to a provided image

    Example:
        "http://app/image?url=https://image/image.png"

    Returns:
        An annotated version of the input URL source with bounding boxes
        for classifications
    """
    url = request.args.get("url")

    # Get the image and evaluate it
    img = get_image(url)

    # Validate the image had been return correctly
    if img is None:
        return Response(
            '"error": "Unable to parse URL to image. Is the endpoint an image and does it exist?',
            status=400,
            mimetype="application/json",
        )

    img = classify_image(img)

    # Serve the generated response
    return serve_pil_image(img)


@app.route("/image_raw")
def image_raw():
    """Evaluate and export an image and it's labels as raw text

    Example:
        "https://app/image_raw?url=https://image/image.png"

    Returns:
        A JSON response containing label information for an image
    """
    url = request.args.get("url")

    # Get the image and evaluate it
    raw_data = get_image(url)

    if raw_data is None:
        return Response(
            '"error": "Unable to parse URL to image. Is the endpoint an image and does it exist?',
            status=400,
            mimetype="application/json",
        )

    raw_data = classify_image_raw(raw_data)

    return raw_data


@app.route("/image_ocr")
def image_ocr() -> str:
    url = request.args.get("url")

    # Get the image
    img = get_image(url)

    if img is None:
        return Response(
            '"error": "Unable to parse URL to image. Is the endpoint an image and does it exist?',
            status=400,
            mimetype="application/json",
        )

    text = evaluate_image(img)

    return text


@app.route("/combined", methods=["POST"])
def image_combined():
    """Evaluate a JSON POST request list of URLs and perform AI object labelling

    Example JSON POST body:
        {
            urls: [
                "https://image/image.jpg",
                "https://image/another.png"
            ]
        }

    Returns:
       A single image/jpeg containing AI object detection annotations
    """
    urls = request.json["urls"]
    images = [get_image(url) for url in urls]

    # Filter out the failed images
    images = list(filter(lambda x: x is not None, images))

    # Combine the images
    combined_img = grid_images(images)

    return serve_pil_image(combined_img)


@app.route("/combined_raw", methods=["POST"])
def image_combined_raw():
    """Evaluate a JSON POST request list of URLs and perform AI object labelling

    Example JSON POST body:
        {
            urls: [
                "https://image/image.jpg",
                "https://image/another.png"
            ]
        }

    Returns:
       A single image/jpeg containing AI object detection annotations
    """
    urls = request.json["urls"]
    images = [get_image(url) for url in urls]

    # Filter out the failed images
    images = list(filter(lambda x: x is not None, images))

    dataset = execute_with_threadpool(images, classify_image_raw)

    return dataset


@app.route("/combined_ocr", methods=["POST"])
def combined_ocr():
    """Evaluate a JSON POST request list of URLs and perform AI OCR recognition

    Example JSON POST body:
        {
            urls: [
                "https://image/image_with_some_text.jpg",
                "https://image/another_with_some_text.png"
            ]
        }

    Returns:
       A single image/jpeg containing AI OCR recognition resolved text
    """
    urls = request.json["urls"]
    images = [get_image(url) for url in urls]

    # Filter out the failed images
    images = list(filter(lambda x: x is not None, images))

    dataset = execute_with_threadpool(images, evaluate_image)

    return dataset


# Enables serving a PIL (pillow image) as a endpoint response
def serve_pil_image(pil_img):
    """Converts a PIL (pillow) image to a HTTP image/jpeg response

    Args:
        pil_img (Image): The PIL image to convert

    Returns:
        A HTTP response that serves the image
    """
    img_io = BytesIO()
    pil_img.save(img_io, "JPEG", quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")
