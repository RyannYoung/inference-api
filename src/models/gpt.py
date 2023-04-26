from models.base import ModelBase
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Dict
from transformers import pipeline
import torch


class Gpt(ModelBase):
    def alias(self) -> str:
        return "Gpt"

    def description(self) -> str:
        return "GPT"

    def load(self) -> None:
        self.image_to_text = pipeline(
            "image-to-text", model="nlpconnect/vit-gpt2-image-captioning"
        )

    def classify_image_raw(self, img: PILImage) -> list[Dict]:
        return {"msg": self.image_to_text(img)["generated_text"][0]}

    def classify_image(self, img: PILImage) -> PILImage:
        return None
