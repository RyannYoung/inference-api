from models.base import ModelBase
from PIL.Image import Image as PILImage
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Dict


class Blip(ModelBase):

    # Has the model loaded
    loaded = False

    # Loaded elements
    processor = None
    model = None

    def load(self) -> None:
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )

    def description(self) -> str:
        return "BLIPPPP"

    def alias(self) -> str:
        return "Blip"

    def classify_image(self, img: PILImage) -> PILImage:
        return super().classify_image(img)

    def classify_image_raw(self, img: PILImage) -> list[Dict]:

        # unconditional image captioning
        inputs = self.processor(img, return_tensors="pt")

        out_tensor = self.model.generate(**inputs)

        return {"msg": self.processor.decode(out_tensor[0], skip_special_tokens=True)}
