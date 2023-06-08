import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from veggideas.recipes import get_recipes
import cv2
import numpy as np

#uvicorn veggideas.api.fast:app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model= tf.keras.models.load_model('models/my_model.h5')

@app.get("/")
def root():
    # $CHA_BEGIN

    return dict(greeting="Hello")
    # $CHA_END

@app.post('/predict')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    cv2_img = np.expand_dims(cv2_img, axis=0)
    breakpoint()
    prediction = app.state.model.predict(cv2_img)
    preds_classes = np.argmax(prediction, axis=-1)[0]
    #LABELS

    #vegetable_type = get_predicted_vegetable(prediction)

    return {"prediction": preds_classes}



def get_predicted_vegetable(prediction):
    return "carrot"
