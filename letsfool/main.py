from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import for validation of user input
from pydantic import BaseModel, Field, field_validator

import base64
import numpy as np
import captum
import io
import os
from PIL import Image

from mnist_model import MNISTModel, bytes_to_tensor


# Set limits for user inputs
MAX_IMAGE_DIM = 280
MAX_PAYLOAD_SIZE = 100000 # Translates to ~75KB of base64 data

class ImageRequest(BaseModel):
    """ImageRequest class that ensures a request involving an
    image is valid by checking the input.
    """
    image: str = Field(max_length=MAX_PAYLOAD_SIZE)

    @field_validator("image")
    @classmethod
    def validate_image(cls, input_data: str) -> str:
        """Validate that input data is a PNG image
        
        Args:
            input_data (str): The Base64 input data
        
        Returns:
            str: The validate input data
        
        Raises:
            ValueError: If the data is not in a valid PNG format
        """
        # Check that it's a PNG data URL (reject anything that isn't)
        if not input_data.startswith("data:image/png;base64,"):
            raise ValueError("User input must be a PNG data URL")

        # First check passed! Next get the base64 data
        _, encoded = input_data.split(",", 1)

        # Validate that base64 is well formed
        try:
            binary_data = base64.b64decode(encoded, validate=True)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        
        # Validate it's actually a PNG by checking for specific bytes
        if not binary_data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError("Data is not a valid PNG")
        
        # Validate dimensions
        try:
            image = Image.open(io.BytesIO(binary_data))
            w, h = image.size
            if w != MAX_IMAGE_DIM or h != MAX_IMAGE_DIM:
                raise ValueError("Image dimensions exceed maximum size {}x{} > {}x{}".format(w, h, MAX_IMAGE_DIM, MAX_IMAGE_DIM))
            if w < 1 or h < 1:
                raise ValueError("Image too small")
        except ValueError:
            raise
        except Exception:
            raise ValueError("Could not parse image")

        return input_data


# Initialise the application
app = FastAPI()

# Give access to the static html files
app.mount("/static", StaticFiles(directory='static', html=True), name='static')

# Set origin as an environment variable to keep things flexible
# Multiple origins can be specified with comma separation
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", default="http://localhost:8000").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

@app.on_event("startup")
def initialise():
    """Initialises the application by creating models that will be used
    throughout the lifetime of the application.
    """
    global model
    model = MNISTModel()
    model.load_model()
    model.eval()
    
    # Set up the explanation using integrated gradients 
    global ig_model
    ig_model = captum.attr.IntegratedGradients(model)



@app.get("/", response_class=HTMLResponse)
def get_frontend():
    """Gets the static html page allowing us to render it"""
    with open("static/index.html", 'r') as file:
        return file.read()


@app.post("/classify")
async def classify_image(request: ImageRequest) -> JSONResponse:
    """Classifies an input image (after verification) when the user makes a request
    from the web page.

    Args:
        request (ImageRequest): The Base64 encoded data corresponding to a PNG
    
    Returns:
        JSONResponse: Returns the response to the web front end in JSON format which includes
        the labels, softmax prediction values and explanation heat map
    """
    image_data = request.image
    _, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image = bytes_to_tensor(binary_data).unsqueeze(dim=0)
    result = model.output_predict(image).squeeze()
    pred_label = int(np.argmax(result))
    
    labels = [str(x) for x in range(10)]

    # Explain the results using integrated gradients
    ig_attr = ig_model.attribute(image, target=pred_label, n_steps=50)
    ig_attr_img = ig_attr[0].permute(1,2,0).squeeze().numpy()
    ig_attr_img = ig_attr_img[::-1, :]
    
    return JSONResponse(content={"labels": labels, "values": result.tolist(), "igimage": ig_attr_img.tolist()})