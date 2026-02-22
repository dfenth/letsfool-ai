from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import base64
import numpy as np
import captum

from mnist_model import MNISTModel, bytes_to_tensor


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
async def classify_image(request: Request):
    data = await request.json()
    image_data = data["image"]
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