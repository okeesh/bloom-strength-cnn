import json
import os
from keras.utils import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Define the paths for the numpy files
train_images_np_file = 'train_images.npy'
train_labels_np_file = 'train_labels.npy'
validation_images_np_file = 'validation_images.npy'
validation_labels_np_file = 'validation_labels.npy'
class_weights_np_file = 'class_weights.npy'

# Check if the numpy files exist
if os.path.exists(class_weights_np_file) and os.path.exists(train_images_np_file) and os.path.exists(
        train_labels_np_file) and os.path.exists(validation_images_np_file) and os.path.exists(
        validation_labels_np_file):
    # Load the data from the numpy files
    print("Loading saved values..")
    train_images = np.load(train_images_np_file)
    train_labels = np.load(train_labels_np_file)
    validation_images = np.load(validation_images_np_file)
    validation_labels = np.load(validation_labels_np_file)
    class_weights = np.load(class_weights_np_file, allow_pickle=True).item()

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
                image_path = next(
                    (os.path.join("dataset/images", image_info["file_name"]) for image_info in images_info if
                     image_info["id"] == image_id), None)

                if image_path is not None:
                    image_paths.append(image_path)
                    labels.append(int(bloom_strength))  # Convert bloom_strength to integer


    # Load images and preprocess them
    def load_and_preprocess_image(image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        return image


    images = [load_and_preprocess_image(image_path) for image_path in image_paths]
    images = np.array(images)
    labels = np.array(labels) - 1  # Subtract 1 from labels to ensure they are in the range 0 to 8

    print(f"Loaded {len(images)} images with corresponding labels")

    from keras.utils import to_categorical

    # ... (previous code remains the same)

    # Split the data into train and validation sets using stratified sampling
    train_images, validation_images, train_labels, validation_labels = train_test_split(images, labels, test_size=0.2,
                                                                                        stratify=labels, random_state=42)

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=9)
    validation_labels = to_categorical(validation_labels, num_classes=9)

    # Save the train and validation data as numpy files
    np.save(train_images_np_file, train_images)
    np.save(train_labels_np_file, train_labels)
    np.save(validation_images_np_file, validation_images)
    np.save(validation_labels_np_file, validation_labels)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(train_labels, axis=1)),
                                                      y=np.argmax(train_labels, axis=1))
    class_weights = dict(enumerate(class_weights))

    # Print out the class weights for each class and the number of images for that respective class
    for i in range(9):
        print(
            f"Class {i} - Number of images: {np.sum(np.argmax(train_labels, axis=1) == i)}, Class weight: {class_weights[i]}")

    np.save(class_weights_np_file, class_weights)