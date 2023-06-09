import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from veggideas.recipes import get_recipes
import cv2
import numpy as np
from veggideas.registry import load_model
from veggideas.recipes import get_recipes, get_recipes_details, make_requests

# uvicorn veggideas.api.fast:app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model = load_model()
print("Model loaded")

@app.get("/")
def root():

    return dict(greeting="Hello")


@app.post('/predict')
async def receive_image(img: UploadFile=File(...)):

    vegg_list = ['Bean', 'Broccoli',
                 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    ### Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    cv2_img = tf.image.resize(cv2_img, (224,224))
    cv2_img = np.expand_dims(cv2_img, axis=0)



    prediction = app.state.model.predict(cv2_img)
    preds_classes = np.argmax(prediction, axis=-1)[0]
    accuracy = round(np.max(prediction, axis=-1)[0]*100,4)

    final_prediction = vegg_list[preds_classes].lower()

    #vegetable_type = get_predicted_vegetable(prediction)

    df = get_recipes_details(10, final_prediction)
    return final_prediction, accuracy, df.to_dict()
