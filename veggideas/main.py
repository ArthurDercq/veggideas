import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from veggideas.registry import load_model
from veggideas.recipes import get_recipes_details


def preprocess_new_image(path):
    image_tf = tf.image.decode_image(tf.io.read_file(path))
    resized_image = tf.image.resize(image_tf, (224,224))
    final_image = tf.expand_dims(resized_image, axis=0)

    print("New image preprocessed correclty ✅")

    return final_image



if __name__ == '__main__':

    image_path = input("Where is your image located? \n")

    new_image = preprocess_new_image(image_path)

    model = load_model()

    prediction = model.predict(new_image)

    vegg_list = ['Bean', 'Broccoli','Cabbage', 'Capsicum',
                 'Carrot', 'Cauliflower', 'Cucumber',
             'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    pred_class = np.argmax(prediction, axis=-1)[0]

    final_prediction = vegg_list[pred_class].lower()

    print(final_prediction)
    # final_recipes = get_recipes_details(3,final_prediction)

    # print(f"You want to find recipes with {final_prediction} ? Here are some: ")

    # print(final_recipes)
