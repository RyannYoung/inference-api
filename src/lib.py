from typing import Dict, Tuple
from PIL import Image
from PIL.Image import Image as PILImage
import requests
from io import BytesIO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import torch
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image, to_tensor
import uuid
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def execute_with_threadpool(lst: list, fn):
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        results = []

        # Create futures and execute
        for item in lst:
            futures.append(executor.submit(fn, item))

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

    # Guard for if the URL is falsey
    if not url:
        return None

    # Initiate the request
    res = requests.get(url)

    try:
        generated_image = Image.open(BytesIO(res.content))

        # Guard for if the image is too small (i.e., favicons)
        if generated_image.width < 50 or generated_image.height < 50:
            return None

        return generated_image
    except:
        print(f"Unable to process image: {url}")
        return None


def process(img: PILImage) -> Tuple:
    """Run an PIL image through a pre-defined model returning
    a tuple containing the prediction, labels, and img tensor
    respectively

    Args:
        img (Image): The PIL image to process into the CNN

    Returns:
        Tuple: Containing the prediction, labels, and img tensor (respectively)
    """

    # Step 0: Create a temporary file (... this is dodgy)
    temp_file = f"{uuid.uuid4()}.jpg"
    img.convert("RGB").save(temp_file)

    img_tensor = read_image(temp_file)  # <- This needs to be a uint8 list

    # Clean it up
    os.remove(temp_file)

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img_tensor)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]

    return (prediction, labels, img_tensor)


def classify_image(img: PILImage) -> PILImage:
    """Generates and returns a PIL Image that has been
    classified and bounding boxed annotated with labels

    Args:
        img (Image): The PIL Image to evaluate

    Returns:
        Image: The annotated PIL Image (run through the CNN)
    """

    # Run the image through the CNN returning a tuple
    (prediction, labels, img) = process(img)

    # Modify the image to include bounding box information of
    # resolved labels
    box = draw_bounding_boxes(
        img,
        boxes=prediction["boxes"],
        labels=labels,
        colors="red",
        width=2,
        font="Poppins-Medium.ttf",
        font_size=16,
    )

    # Return the box as a PIL Image
    return to_pil_image(box.detach())


def classify_image_raw(img: PILImage) -> Dict:
    """Run a PIL Image through a CNN label classifier
    returning a python DICT containing the resolved label
    classifications

    Args:
        img (Image): The PIL image to evaluate

    Returns:
        Dict: The label classification in text format
    """

    # Run the image through the CNN returning a tuple
    (prediction, labels, _) = process(img)

    # Generate a list based on the classifications
    boxes = []
    for (i, box) in enumerate(prediction["boxes"]):
        output = {
            "type": labels[i],
            "position": {
                "x": box[0].item(),
                "y": box[1].item(),
                "w": box[2].item(),
                "h": box[3].item(),
            },
        }

        boxes.append(output)

    # Return the data
    return boxes


def grid_images(imgs: list[PILImage]) -> PILImage:
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
        img = classify_image(pil_img)

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


if __name__ == "__main__":
    """Runs a handful of tests when the script is executed directly"""

    horse_img = Image.open("assets/horse.jpg")
    giraffe_img = Image.open("assets/giraffe.jpg")
    pelican_img = Image.open("assets/pelican.jpg")

    imgs = [horse_img, giraffe_img, pelican_img]

    combined_img = grid_images(imgs)

    combined_img.save("combined.jpg")
