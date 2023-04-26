from models.base import ModelBase
from torchvision import transforms
from PIL.Image import Image
import torch
from torchvision import transforms


class AlexNet(ModelBase):
    def load(self) -> None:
        """Use the init function to load the model

        Storing the model value should drastically increase
        the model load speeds for each call.

        """

        # Load the Model
        model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
        model.eval()

        # Define the preprocess
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = model
        self.preprocess = preprocess

    def alias(self) -> str:
        return "AlexNet"

    def description(self) -> str:
        return "AlexNet model"

    def classify_image(self, img: Image) -> Image:
        return super().classify_image(img)

    def classify_image_raw(self, img: Image) -> list:
        input_tensor = self.preprocess(img)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)  # type: ignore

        with torch.no_grad():
            output = self.model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Output labels
        labels = []

        # Read the categories
        with open("assets/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            labels.append((categories[top5_catid[i]], top5_prob[i].item()))

        return labels
