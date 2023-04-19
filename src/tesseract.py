import pytesseract
from PIL import Image
from PIL.Image import Image as PILImage
import uuid
import subprocess
from tempfile import TemporaryDirectory


def clean_image(img: PILImage) -> PILImage:
    """Cleans an image attempting to make the image more
    visible to OCR increasing text accuracy.

    Note: Temporarily creates a file to process
    Note: This does can sometimes result in poorer accuracy
    Note: Requires ImageMagick to be installed
    Note: Requires textcleaner (http://www.fmwconcepts.com/imagemagick/textcleaner/index.php)
    to be accessible from PATH

    Args:
        img (PILImage): The Image to process

    Returns:
        PILImage: The cleaned version of the image
    """

    with TemporaryDirectory() as tmpdir:
        print(tmpdir)

        # Create a temporary file with the image
        temp_file = f"{tmpdir}/{uuid.uuid4()}.png"

        # Generate a temp file
        img.save(temp_file)

        # Clean the file
        # Note: This feature only works if you have BOTH
        #   1. ImageMagick Installed
        #   2. textcleaner installed in path
        #
        # If not installed, these features will not run
        cmd = f"textcleaner {temp_file} {temp_file}"
        subprocess.run(cmd, shell=True)

        # Re-import it
        img = Image.open(temp_file)

    # Return it
    return img


def evaluate_image(img: PILImage, clean: bool = False) -> str:
    """Reads text from a PIL Image

    Args:
        img (PILImage): The PIL Image to parse
        clean (bool, optional): Whether to further clean the image for OCR. Defaults to False.

    Returns:
        str: The resolved text
    """

    # Optionally clean the image
    if clean:
        img = clean_image(img)

    return pytesseract.image_to_string(img).strip()


if __name__ == "__main__":
    # Basic OCR Test with cleaning
    image = Image.open("clifford.jpg")
    text = evaluate_image(image, True)
    print(text)
