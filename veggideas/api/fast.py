import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from veggideas.recipes import get_recipes
import cv2
import numpy as np
from veggideas.registry import load_model

# uvicorn veggideas.api.fast:app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model= load_model()

@app.get("/")
def root():
    # $CHA_BEGIN

    return dict(greeting="Hello")
    # $CHA_END

@app.post('/predict')
async def receive_image(img: UploadFile=File(...)):

    vegg_list = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    cv2_img = np.expand_dims(cv2_img, axis=0)
    prediction = app.state.model.predict(cv2_img)
    preds_classes = np.argmax(prediction, axis=-1)[0]
    #LABELS

    temp_dict = dict(zip(preds_classes, vegg_list))
    final_prediction = temp_dict[1]

    #vegetable_type = get_predicted_vegetable(prediction)

    return {"prediction": final_prediction}



def get_predicted_vegetable(prediction):
    return "carrot"
