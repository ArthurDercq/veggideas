import requests

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
    URL = f"https://api.edamam.com/api/recipes/v2?type=public&q={vegetable_type}&app_id={APP_ID}&app_key={API_KEY}&diet={DIET}&calories={CALORIES}&field=url&field=dietLabels&field=healthLabels&field=ingredients&field=calories&field=cuisineType&field=mealType"

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
