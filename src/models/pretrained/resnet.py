from typing import Dict, Tuple
import uuid
from torch import Tensor
from models.base import ModelBase
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.io.image import read_image
import os
from PIL.Image import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import ImageDraw, ImageFont


class Resnet(ModelBase):
    def load(self) -> None:
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def alias(self) -> str:
        return "ResNet"

    def description(self) -> str:
        return "The resnet model"

    def classify(self, img: Image) -> Tuple[Dict, list, Tensor]:
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.8
        )[0]

        formatted_results = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            result = {
                "box": [round(i, 2) for i in box.tolist()],
                "label": self.model.config.id2label[label.item()],
                "score": round(score.item(), 3),
            }

            formatted_results.append(result)

        return formatted_results

    def classify_image(self, img: Image) -> Image:
        # Run the image through the CNN returning a tuple
        results = self.classify(img)

        draw = ImageDraw.Draw(img)
        for result in results:
            # Draw bounding box information
            draw.rectangle(result["box"], outline="red", width=2)

            # Add the label and score
            padding = 4
            x_text = result["box"][0] + padding
            y_text = result["box"][1] + padding
            draw.text(
                (x_text, y_text),
                f'{result["label"].upper()} (score: {result["score"]})',
                font=ImageFont.truetype("Poppins-Medium.ttf", 32),
                fill=(255, 0, 0, 255),
            )

        # Return the box as a PIL Image
        return img

    def classify_image_raw(self, img) -> list:
        # Get the image data
        return self.classify(img)
