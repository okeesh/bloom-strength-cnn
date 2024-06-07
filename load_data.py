import json
import os
from keras.utils import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
import numpy as np

# Define the paths for the numpy files
train_images_np_file = 'train_images.npy'
train_labels_np_file = 'train_labels.npy'
test_images_np_file = 'test_images.npy'
test_labels_np_file = 'test_labels.npy'

# Check if the numpy files exist
if os.path.exists(train_images_np_file) and os.path.exists(train_labels_np_file) and os.path.exists(test_images_np_file) and os.path.exists(test_labels_np_file):
    # Load the data from the numpy files
    print("Loading saved values..")
    train_images = np.load(train_images_np_file)
    train_labels = np.load(train_labels_np_file)
    test_images = np.load(test_images_np_file)
    test_labels = np.load(test_labels_np_file)
else:

    # Load the annotation file
    annotation_file = "dataset/annotations/instances_Train.json"
    with open(annotation_file) as f:
        coco_data = json.load(f)

    images_info = coco_data["images"]
    categories = coco_data["categories"]
    annotations_info = coco_data["annotations"]

    image_paths = []
    labels = []

    for annotation in annotations_info:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        # Find the corresponding category name
        category_name = next((category["name"] for category in categories if category["id"] == category_id), None)

        # Check if the category name is "tree"
        if category_name == "tree":
            # Extract the "bloom_strength" attribute value
            attributes = annotation["attributes"]
            bloom_strength = attributes.get("bloom strenght")

            if bloom_strength is not None:
                # Find the corresponding image path
                image_path = next((os.path.join("dataset/images", image_info["file_name"]) for image_info in images_info if image_info["id"] == image_id), None)

                if image_path is not None:
                    image_paths.append(image_path)
                    labels.append(bloom_strength)

    # Load images and preprocess them
    def load_and_preprocess_image(image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        return image

    images = [load_and_preprocess_image(image_path) for image_path in image_paths]
    images = np.array(images)

    print(f"Loaded {len(images)} images with corresponding labels")

    # Create a list of tuples containing image paths and corresponding labels
    data = list(zip(image_paths, labels))

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create separate lists for train and test images and labels
    train_images_paths, train_labels = zip(*train_data)
    test_images_paths, test_labels = zip(*test_data)

    # Load train and test images
    train_images = np.array([load_and_preprocess_image(img_path) for img_path in train_images_paths])
    test_images = np.array([load_and_preprocess_image(img_path) for img_path in test_images_paths])

    # Subtract 1 from labels to ensure they are in the range 0 to 8
    train_labels = np.array([label - 1 for label in train_labels])
    test_labels = np.array([label - 1 for label in test_labels])

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=9)
    test_labels = to_categorical(test_labels, num_classes=9)

    np.save(train_images_np_file, train_images)
    np.save(train_labels_np_file, train_labels)
    np.save(test_images_np_file, test_images)
    np.save(test_labels_np_file, test_labels)