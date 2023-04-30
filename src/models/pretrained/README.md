# Auto-loading Pre-trained models

The `Model Controller` enables automatic importing of models through its `auto_load()` function. This function makes it easy to write, and add a new model to use in the application.

## Creating a new model

To create a new model and load it into the controller, follow these steps:

1. Create a new model file in the src/models/pretrained directory (must end in `.py`)
2. Create a class the extends from the ModelBase class

```python
from models.base import ModelBase
from PIL.Image import Image as PILImage

class Newmodel(ModelBase):
    def alias(self) -> str:
        return "NewModel"

    def description(self) -> str:
        return "My brand new model"

    def load(self) -> None:
        # Define loading logic here

    def classify_image_raw(self, img: PILImage) -> dict:
        # Define text-based responses

    def classify_image(self, img: PILImage) -> PILImage:
        # Define image-based responses (Pillow image)
```

... and thats its. The model controller will handle the loading, API calls and display of description/alias

<b>Note: The class name of the model needs to be captilalised (specifically, first-letter uppercase and rest lowercase)</b>
