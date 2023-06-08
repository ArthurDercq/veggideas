import requests
import sqlite3
import json
import urllib
from random import randint

APP_ID = "785b538a"
API_KEY = "01bde07232a776b52909d9876ae11db4"
VEGETABLE = ["carrot", "chicken"]
DIET = "high-fiber"
CALORIES = "100-800"
selected_health_labels= ['Vegetarian']


URL = f"https://api.edamam.com/api/recipes/v2?type=public&q={VEGETABLE}&app_id={APP_ID}&app_key={API_KEY}&diet={DIET}&calories={CALORIES}&field=url&field=dietLabels&field=healthLabels&field=ingredients&field=calories&field=cuisineType&field=mealType"

def make_requests(url):
    response = requests.get(url)
    data = response.json()
    return data

recipes = make_requests(URL)


def filter_healthlabels_new(recipes, health_labels):
    return [hit['recipe'] for hit in recipes['hits']
            if all(label.capitalize() in hit['recipe']['healthLabels'] for label in health_labels)]



def filter_by_ingredient_count(recipes, min_ingredients, max_ingredients):
    filtered_recipes = []

    for recipe in filter_healthlabels_new(recipes, selected_health_labels):
        ingredients = recipe['ingredients']
        num_ingredients = len(ingredients)

        if min_ingredients <= num_ingredients <= max_ingredients:
            filtered_recipes.append(recipe)

    return filtered_recipes
