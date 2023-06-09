import requests
import pandas as pd
import re



APP_ID = "785b538a"
API_KEY = "01bde07232a776b52909d9876ae11db4"
DIET = "high-fiber"
CALORIES = "100-800"
selected_health_labels = ['Vegetarian']

def make_requests(url):
    response = requests.get(url)
    data = response.json()
    return data

def get_recipes(vegetable_type):
    URL = f"https://api.edamam.com/api/recipes/v2?type=public&q={vegetable_type}&app_id={APP_ID}&app_key={API_KEY}&diet={DIET}&calories={CALORIES}&field=label&field=url&field=dietLabels&field=healthLabels&field=ingredients&field=calories&field=cuisineType&field=mealType&field=totalTime"

    recipes = make_requests(URL)
    return recipes

def filter_healthlabels_new(recipes, health_labels):
    return [hit['recipe'] for hit in recipes['hits']
            if all(label.capitalize() in hit['recipe']['healthLabels'] for label in health_labels)]

def filter_by_ingredient_count(recipes, min_ingredients, max_ingredients):
    filtered_recipes = []
    excluded_foodcategories = ["Oils", "Condiments and sauces"]
    for hit in recipes["hits"]:
        ingredients = (hit["recipe"]["ingredients"])
        num_ingredients = len(ingredients)
        for ingredient in ingredients:
            if ingredient["foodCategory"].lower() in excluded_foodcategories:
                num_ingredients -= 1

        if min_ingredients <= num_ingredients <= max_ingredients:
            filtered_recipes.append(hit)

    return filtered_recipes

def filter_by_time(recipes, max_time):
    return[hit for hit in recipes["hits"] if hit["recipe"]["totalTime"] <= max_time]



def get_recipes_details(number_of_recipes, final_prediction):
    recipe_list = []

    recipes = get_recipes(final_prediction)

    for _ in range(0, number_of_recipes):
        # Name of the recipe
        url = recipes["hits"][_]["recipe"]["url"]
        recipe_name = recipes['hits'][_]['recipe']['label']

        if not recipe_name:  # Skip recipes without a name
            continue

        # Recipe URL
        recipe_url = recipes["hits"][_]["recipe"]["url"]

        # List of ingredients and relative quantity
        list_ingredients = [item["text"] for item in recipes["hits"][_]["recipe"]["ingredients"]]


        # Type of diet
        recipe_type_diet = recipes["hits"][_]["recipe"]["healthLabels"][:3]

        # Calories of the recipe
        recipe_calories = round(recipes["hits"][_]["recipe"]["calories"])

        recipe_time = int(recipes["hits"][_]["recipe"]["totalTime"])

        recipe_meal_type = [meal.capitalize() for meal in recipes["hits"][_]["recipe"]["mealType"][0].split("/")]
        recipe_cuisine_type = recipes["hits"][_]["recipe"]["cuisineType"][0].capitalize()

        # Append recipe details to the list
        recipe_details = {
            "Recipe Name": recipe_name,
            "Ingredients": list_ingredients,
            "Diet Type": recipe_type_diet,
            "Calories": recipe_calories,
            "Time": recipe_time,
            "Meal Type": recipe_meal_type,
            "Cuisine": recipe_cuisine_type,
            "Recipe URL": recipe_url

        }
        recipe_list.append(recipe_details)

    # Convert the list of dictionaries into a dataframe
    df = pd.DataFrame(recipe_list)

    return df
