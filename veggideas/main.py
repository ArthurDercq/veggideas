import cv2
import numpy as np
import pandas as pd
from veggideas.registry import load_model
from veggideas.recipes import get_recipes_details


def preprocess_new_image(path):
# Demande à l'utilisateur de sélectionner une image depuis l'ordinateur
    image_path = "/Users/arthurdercq/Desktop/pumpkin.jpeg"

    # Charge the imahe with OpenCV
    image = cv2.imread(path)
    resized_image = cv2.resize(image, (224, 224))
    resized_image = np.expand_dims(resized_image, axis=0)

    print("New image preprocessed correclty ✅")

    return resized_image



if __name__ == '__main__':

    image_path = input("Where is your image located? \n")

    new_image = preprocess_new_image(image_path)

    model = load_model()

    prediction = model.predict(new_image)

    vegg_list = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
             'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
             'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    pred_class = np.argmax(prediction, axis=-1)[0]

    final_prediction = vegg_list[pred_class].lower()

    final_recipes = get_recipes_details(10,final_prediction)

    print(f"You want to find recipes with {final_prediction} ? Here are some: ")

    print(final_recipes)
