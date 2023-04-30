from models.base import ModelBase
from PIL.Image import Image as PILImage
from PIL import Image, ImageDraw
from typing import Dict
import pytesseract
from pytesseract import Output, TesseractNotFoundError

# REQUIRED PARAMETERS
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


class Tesseract(ModelBase):
    def alias(self) -> str:
        return "Tesseract"

    def description(self) -> str:
        return "Tesseract is an open source text recognition (OCR) Engine, available under the Apache 2.0 license. For more information visit https://tesseract-ocr.github.io/"

    def load(self) -> None:
        # Check if tesseract is installed in path
        try:
            pytesseract.get_tesseract_version()
        except TesseractNotFoundError:
            print(
                "Tesseract is not installed on the system, or accessible from identified PATH"
            )

    def classify_image_raw(self, img: PILImage) -> list[Dict]:
        res = pytesseract.image_to_string(img)
        return res

    def classify_image(self, img: PILImage) -> PILImage:
        res = pytesseract.image_to_data(img, output_type=Output.DICT)

        draw = ImageDraw.Draw(img)
        n_boxes = len(res)

        for i in range(n_boxes):
            (x, y, w, h) = (
                res["left"][i],
                res["top"][i],
                res["width"][i],
                res["height"][i],
            )
            # Draw the bounding box on the image
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        return img
