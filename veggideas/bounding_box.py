import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import selectivesearch
import numpy as np
from veggideas.registry import load_model



def load_image(img_path):
# Load the image
    image_path = img_path
    image = cv2.imread(image_path)
    return image

def create_bounding_boxes(image):

    # Perform selective search to generate candidate regions
    selective_search_results = selectivesearch.selective_search(image, scale=750, sigma=0.9, min_size=100)

    minimum_bounding_box_size = 100
    maximum_aspect_ratio = 3
    # Calculate target aspect ratio range (adjust as per your requirement)
    target_aspect_ratio = 1
    aspect_ratio_range = 0.30

    selected_candidates = []
    for i, candidate in enumerate(selective_search_results[1]):
        x, y, w, h = candidate['rect']

        # Calculate bounding box size and aspect ratio
        bounding_box_size = w * h
        aspect_ratio = max(w / h, h / w)

        # Exclude the bounding box of the whole picture
        image_size = image.shape[0] * image.shape[1]
        box_coverage = bounding_box_size / image_size
        if box_coverage < 0.98 and bounding_box_size >= minimum_bounding_box_size and aspect_ratio <= maximum_aspect_ratio:
            selected_candidates.append({'rect': (x, y, w, h), 'index': i})


    # Function to calculate the coverage ratio between two bounding boxes
    def calculate_coverage_ratio(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection_x = max(x1, x2)
        intersection_y = max(y1, y2)
        intersection_w = min(x1 + w1, x2 + w2) - intersection_x
        intersection_h = min(y1 + h1, y2 + h2) - intersection_y

        intersection_area = max(0, intersection_w) * max(0, intersection_h)
        box1_area = w1 * h1
        box2_area = w2 * h2

        return intersection_area / box1_area, box1_area, box2_area

    # List to store the filtered bounding boxes
    filtered_boxes = []

    # Iterate through the selected bounding boxes
    for i in range(len(selected_candidates)):
        current_box = selected_candidates[i]['rect']
        is_contained = False

        # Check if the coverage ratio of the current box is less than 90% of any other box
        for j in range(len(selected_candidates)):
            if i != j:
                other_box = selected_candidates[j]['rect']
                coverage_ratio, current_area, other_area = calculate_coverage_ratio(current_box, other_box)
                if coverage_ratio >= 1:
                    # The current box is contained within another box
                    is_contained = True
                    if current_area >= other_area:
                        filtered_boxes.append(current_box)
                    else:
                        break
        # If the current box is not contained, add it to the filtered list
        if not is_contained:
            filtered_boxes.append(current_box)


    unique_tuples = set(filtered_boxes)
    filtered_list = list(unique_tuples)

    list_subimages =[]

    # iterate over filtered coordinates
    for coords in filtered_list:

        x, y, w, h = coords

        #Put it in the right format for the model
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (224, 224))
        final_image = np.expand_dims(resized_image, axis=0)
        list_subimages.append(final_image)

    return list_subimages

    # # # Create a figure and axes
    # fig, ax = plt.subplots()

    # # Plot the image
    # ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # rect = []
    # for i, box in enumerate(filtered_list):
    #     x, y, w, h = box
    #     rect.append(patches.Rectangle((x, y), w, h, edgecolor='green', facecolor='none', linewidth=2))
    #     ax.add_patch(rect[i])

    #     aspect_ratio = max(w / h, h / w)
    #     ax.text(x, y, f'Box{i}: {round((aspect_ratio - target_aspect_ratio) * 100, 2)}%', color='green')

    # # # Show the figure
    # plt.show()

def predict_bboxes(subimages_list):

    vegg_list = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    model = load_model()

    predictions = []

    for image in subimages_list:
        prediction = model.predict(image)

        pred_class = np.argmax(prediction, axis=-1)[0]

        final_prediction = vegg_list[pred_class].lower()

        predictions.append(final_prediction)

    return predictions



if __name__ == '__main__':

#'/Users/arthurdercq/Desktop/capsicums.jpeg'

    im_path = input("Where is your image located? \n")

    image = load_image(im_path)

    subimages = create_bounding_boxes(image)

    final_predictions = predict_bboxes(subimages)

    print(final_predictions)
