from multiprocessing.dummy import Pool as ThreadPool

from models.base import ModelBase
from models.resnet import ResNet
from models.alexnet import AlexNet
from models.vit import Vit
from models.gpt import Gpt
from models.blip import Blip
from models.tesseract import Tesseract

# Define models to pre-load
preloaded_models: list[ModelBase] = [
    ResNet(),
    AlexNet(),
    Vit(),
    Gpt(),
    Blip(),
    Tesseract(),
]


class ModelController:
    """Handles the management of a set of models"""

    # The models to control
    models: list = []

    def __init__(self) -> None:
        self.models = preloaded_models

    def find_model(self, input: str) -> ModelBase | None:
        """Resolves a model type from an input string.
        This is resolved based on the value returned from the
        models alias() method

        Args:
            input (str): The input to match

        Returns:
            ModelBase | None: The first-matched model (or None if no result)
        """
        matched_model = None
        for model in self.models:
            matcher = input.lower()
            target = model.alias().lower()

            if matcher == target:
                matched_model = model
        return matched_model

    def display_models(self) -> list:
        """Returns a descriptive string about each of the available models

        Returns:
            str: A descriptive string about each of the available models
        """

        models = []

        for model in self.models:
            models.append(
                {"name": model.alias().lower(), "description": model.description()}
            )

        return models

    def display_available_models(self) -> str:
        """Returns a list of the available models via their alias"""
        available_models = [model.alias() for model in self.models]
        return " ".join(available_models)

    def load_models(self, num_threads: int = 8) -> None:
        """Loads all models using threading"""
        pool = ThreadPool(num_threads)
        pool.map(lambda model: model.load(), self.models)
