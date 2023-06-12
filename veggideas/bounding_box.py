import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import selectivesearch


def draw_bboxes():
    image_path = '/Users/arthurdercq/Desktop/eggplants.jpeg'

    # Load the image
    image = cv2.imread(image_path)

    # Perform selective search to generate candidate regions
    selective_search_results = selectivesearch.selective_search(image, scale=750, sigma=0.9, min_size=100)

    minimum_bounding_box_size = 100
    maximum_aspect_ratio = 3
    # Calculate target aspect ratio range (adjust as per your requirement)
    target_aspect_ratio = 0.30
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
                if coverage_ratio >= 0.99:
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
    # Itérer sur les coordonnées filtrées
    for coords in filtered_list:

        x, y, w, h = coords

        # Extraire l'image à partir des coordonnées
        cropped_image = image[y:y+h, x:x+w]

        list_subimages.append(cropped_image)

    return list_subimages

if __name__ == '__main__':
    su_images = draw_bboxes()
