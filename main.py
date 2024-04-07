import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "*",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:52152",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = tf.keras.models.load_model("./Models/model_version_1")
CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

def read_file_as_image(data, target_size=(300, 300)) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize(target_size)
    image = np.array(image)
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction[0])

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
