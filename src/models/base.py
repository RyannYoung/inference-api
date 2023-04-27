from abc import ABC, abstractmethod
from PIL.Image import Image
from typing import Dict

"""Available model modules for use within the inference api"""


class ModelBase(ABC):
    """An abstract class that all models must derive from.
    All methods need to be implemented to correctly execute
    an individuals model

    Args:
        ABC (Abstract Base Class): The abstract base class python identifier (... just ignore)
    """

    @property
    @abstractmethod
    def alias(self) -> str:
        """The str identifier for the model. Typically used to find a model
        from within a ModelController object

        Returns:
            str: The alias for the model controller
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Descriptive text relating to what the model is
        and how to use it.

        Returns:
            str: The models descriptive text
        """
        pass

    def execute(
        self, mode: str, img: Image, format: str | Image | list
    ) -> str | Image | list:
        """The main execution function which is called by the inference-api
        This function should take an image tensor, and run it with the
        invoked model.

        Args:
            img (Image): The image tensor

        Returns:
            Tuple[Dict, list, Tensor]: A tuple containing the prediction, labels, and tensor
        """
        match mode:
            case "classify":
                if format == "text":
                    return self.classify_image_raw(img)

                if format == "image":
                    # Get the data from the image
                    return self.classify_image(img)

                if format == "default":
                    return self.classify_image(img)

        return "Unable to execute a specified mode with input params"

    @abstractmethod
    def load(self) -> None:
        """Load the model, and any other preprocess information"""
        pass

    @abstractmethod
    def classify_image(self, img: Image) -> Image:
        """Classify an image using the model

        Args:
            img (Image): The image to classify (as a PIL image)

        Returns:
            Image: A classified PIL image (i..e, )
        """
        return None
        return "No Image classification has been defined for this model. Try setting 'format' to 'json'"

    @abstractmethod
    def classify_image_raw(self, img: Image) -> list[Dict]:
        """Classify an image using the model and return raw/text data

        Args:
            img (Image): The image to classify

        Returns:
            list[Dict]: A list containing the response classification
        """
        return None
        return "No raw classification has been defined for this model. Try setting 'format' to 'img'"
