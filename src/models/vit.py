from models.base import ModelBase
from PIL.Image import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import torch


class Vit(ModelBase):
    def alias(self) -> str:
        return "Vit"

    def description(self) -> str:
        return "VIT Description"

    def load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def classify_image_raw(self, img: Image) -> list:

        inputs = self.processor(img, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()

        return {"msg": self.model.config.id2label[predicted_label]}

    def classify_image(self, img: Image) -> Image:
        return super().classify_image(img)
