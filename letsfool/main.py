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
from PIL import Image

from mnist_model import MNISTModel, bytes_to_tensor


# Set limits for user inputs
MAX_IMAGE_DIM = 280
MAX_PAYLOAD_SIZE = 100000 # Translates to ~75KB of base64 data

class ImageRequest(BaseModel):
    image: str = Field(max_length=MAX_PAYLOAD_SIZE)

    @field_validator("image")
    @classmethod
    def validate_image(cls, input_data):
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



app = FastAPI()

app.mount("/static", StaticFiles(directory='static', html=True), name='static')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # NOTE: DO NOT USE IN PRODUCTION
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def initialise():
    global model
    model = MNISTModel()
    model.load_model()
    model.eval()
    
    # Set up the explanation using integrated gradients 
    global ig_model
    ig_model = captum.attr.IntegratedGradients(model)



@app.get("/", response_class=HTMLResponse)
def get_frontend():
    with open("static/index.html", 'r') as file:
        return file.read()


@app.post("/classify")
async def classify_image(request: ImageRequest):
    # data = await request.json()
    image_data = request.image #data["image"]
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