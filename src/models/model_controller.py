from importlib import import_module
from multiprocessing.dummy import Pool as ThreadPool

from models.base import ModelBase
from models.pretrained.resnet import Resnet
from models.pretrained.alexnet import Alexnet
from models.pretrained.vit import Vit
from models.pretrained.gpt import Gpt
from models.pretrained.blip import Blip
from models.pretrained.tesseract import Tesseract
from functools import cache

import os

# # Define models to pre-load
# preloaded_models: list[ModelBase] = [
#     ResNet(),
#     AlexNet(),
#     Vit(),
#     Gpt(),
#     Blip(),
#     Tesseract(),
# ]


class ModelController:
    """Handles the management of a set of models"""

    # The models to control
    models: list = []

    def __init__(self) -> None:
        self.models = self.auto_load("src/models/pretrained")

    def auto_load(self, model_dir: str) -> list:
        """Automatically load pretrained models from a specified directory

        Args:
            model_dir (str): The model directory to load from

        Returns:
            list: A list of loaded model
        """

        preloaded_models = []

        # Get a list of all the models
        models = [model for model in os.listdir(model_dir) if model.endswith(".py")]

        for model in models:
            # Get the module name
            module_name = os.path.splitext(model)[0]
            model_module = import_module(f"models.pretrained.{module_name}")

            # Get the model class
            model_class = getattr(model_module, module_name.capitalize())

            # Instantiate the model
            model_instance = model_class()

            # Add the model to the list
            preloaded_models.append(model_instance)

        print(preloaded_models)

        return preloaded_models

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

    @cache
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

    @cache
    def load_models(self, num_threads: int = 8) -> None:
        """Loads all models using threading"""
        pool = ThreadPool(num_threads)
        pool.map(lambda model: model.load(), self.models)


if __name__ == "__main__":
    controller = ModelController()
    print(controller.display_models())
