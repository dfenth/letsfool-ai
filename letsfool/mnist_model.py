import torch
import torchvision
from torchvision.transforms import v2

import numpy as np

from io import BytesIO
from PIL import Image

class AbsModel(torch.nn.Module):
    """An abstract model class"""
    def __init__(self):
        super(AbsModel, self).__init__()

    def load_model(self):
        """Load the model from a specified path"""
        self.load_state_dict(torch.load("mnist_model.pth", weights_only=True, map_location=torch.device('cpu')))


class MNISTModel(AbsModel):
    """The MNIST model class inheriting from AbsModel"""
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = torch.nn.Linear(7*7*64, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes data forward through the model for classification
        
        Args:
            x (torch.Tensor): The input tensor of data to classify
        
        Returns:
            torch.Tensor: The output logits
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.Dropout(p=0.5)(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.Dropout(p=0.5)(x)
        
        x = torch.nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.Dropout(p=0.5)(x)
        logits = self.fc2(x)

        out = logits
        return out
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the class of the input
        
        Args:
            x (torch.Tensor): The input data as a tensor
        
        Returns:
            torch.Tensor: The softmax probabilities of each class
        """
        self.eval()
        out = self.forward(x)
        return torch.softmax(out, dim=1)

    def output_predict(self, x: torch.Tensor) -> np.ndarray:
        """Performs prediction, returning the result as a numpy array
        
        Args:
            x (torch.Tensor): The input to classify
        
        Returns:
            np.ndarray: The softmax results as a numpy array
        """
        res = self.predict(x)
        return res.detach().numpy()


# The input transformation
input_transform = torchvision.transforms.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(28, 28)),
    v2.ToDtype(torch.float32, scale=True), # Scales image to 0,1
])

def bytes_to_tensor(bytes: str) -> torch.Tensor:
    """Converts raw bytes to an image, transforming it so it
    can be processed by the model
    
    Args:
        bytes (str): String of bytes that represent an image
    
    Returns:
        torch.Tensor: The decoded and transformed image
    """
    image = Image.open(BytesIO(bytes)).convert("L")
    image = input_transform(image)
    return image
