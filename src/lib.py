from typing import Dict
from PIL import Image
from PIL.Image import Image as PILImage
import requests
from io import BytesIO
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import send_file
from models.base import ModelBase


# Enables serving a PIL (pillow image) as a endpoint response
def serve_pil_image(pil_img: Image.Image):
    """Converts a PIL (pillow) image to a HTTP image/jpeg response

    Args:
        pil_img (Image): The PIL image to convert

    Returns:
        A HTTP response that serves the image
    """
    pil_img = pil_img.convert("RGB")
    img_io = BytesIO()
    pil_img.save(img_io, "JPEG", quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


def execute_with_threadpool(lst: list, fn, *args):
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        results = []

        # Create futures and execute
        for item in lst:
            futures.append(executor.submit(fn, item, *args))

        for future in as_completed(futures):
            results.append(future.result())

        return results


def get_image(url: str) -> PILImage | None:
    """Convert an URL into a PIL Image

    Args:
        url (str): The URL to attempt to convert

    Returns:
        Image: A PIL image (or None upon error)
    """

    min_height = 50
    min_width = 50

    try:
        # Stole this from many hugging face examples
        img = Image.open(requests.get(url, stream=True).raw)

        if img.width < min_width or img.height < min_height:
            return None

        return img
    except:
        print(f"Unable to process URL into image [{url}]")
        return None


def classify(img: PILImage, model: ModelBase, format: str) -> PILImage | list | str:
    """Wrapper function to run a classification and output to a specified type

    Args:
        img (PILImage): The PIL image to classify
        format (Literal[&quot;default&quot;, &quot;json&quot;]):
        The format to return to

    Returns:
        PILImage | Dict: An image or dict depending on format parameter
    """
    # Check if there is a model available which matches the query
    return model.execute("classify", img, format)


def classify_group(
    imgs: list[PILImage], model: ModelBase, format: str
) -> PILImage | Dict | list:
    """Wrapper function to run a classification on a group of images and output
    to a specified type. See classify() for similar functionality

    Args:
        imgs (List[PILImage]): The list of images to classify
        format (Literal[&quot;default&quot;, &quot;json&quot;]): The output format

    Returns:
        PILImage | Dict: _description_
    """

    match format:
        case "json":
            return execute_with_threadpool(imgs, model.classify_image_raw)
        case _:
            return grid_images(imgs, model)


def grid_images(imgs: list[PILImage], model: ModelBase) -> PILImage:
    """Generates a singular grid image from a list of PIL images
    Note: These images have been evaluated and classified using the
    CNN Classifier

    Args:
        imgs (list[Image]): The list of images to evaluate

    Returns:
        A gridded and evaluated PIL Image
    """

    # Set the square (1:1 aspect ratio) for all images
    SQUARE_DIMENSIONS = 1024

    # Loop through and classify each image
    classified_imgs = []

    for pil_img in imgs:
        # Classify the resized image (returning the img tensor)
        img = model.classify_image(pil_img)

        if img is None:
            return "No Image classification has been defined for this model. Try setting 'format' to 'json'"

        # Resize the images to all be the same
        # NOTE: This will stretch and malform the images
        # Required as all image sizes are to be the same
        # A better method would be to add border (... future implementation)
        resized = img.resize(size=(SQUARE_DIMENSIONS, SQUARE_DIMENSIONS))

        classified_imgs.append(to_tensor(resized))

    # Generate the grid from the tensor data
    grid = make_grid(classified_imgs)

    # Convert back to a PIL Image
    return to_pil_image(grid)
