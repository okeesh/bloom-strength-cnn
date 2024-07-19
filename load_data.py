import json
import os
import re
import numpy as np
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


def delete_saved_data(subdirectory='numpy_files'):
    # Define the paths for the numpy files
    train_images_np_file = os.path.join(subdirectory, 'train_images.npy')
    train_labels_np_file = os.path.join(subdirectory, 'train_labels.npy')
    validation_images_np_file = os.path.join(subdirectory, 'validation_images.npy')
    validation_labels_np_file = os.path.join(subdirectory, 'validation_labels.npy')
    class_weights_np_file = os.path.join(subdirectory, 'class_weights.npy')

    # Delete the numpy files if they exist
    for file_path in [train_images_np_file, train_labels_np_file, validation_images_np_file, validation_labels_np_file,
                      class_weights_np_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")

    print("Saved data deleted successfully.")


def load_data(model_type='classification'):
    # Define the subdirectory for the numpy files
    subdirectory = 'numpy_files'

    # Define the paths for the numpy files
    train_images_np_file = os.path.join(subdirectory, 'train_images.npy')
    train_labels_np_file = os.path.join(subdirectory, 'train_labels.npy')
    validation_images_np_file = os.path.join(subdirectory, 'validation_images.npy')
    validation_labels_np_file = os.path.join(subdirectory, 'validation_labels.npy')
    class_weights_np_file = os.path.join(subdirectory, 'class_weights.npy')

    # Define paths for the new dataset
    new_image_dir = "dataset/images_title_lable"
    new_annotation_file = "dataset/annotations/new_annotations.json"

    # Function to extract bloom strength from filename
    def extract_bloom_strength(filename):
        match = re.search(r'_(\d+\.\d+)_', filename)
        if match:
            bloom_strength = float(match.group(1))
            return int(round(bloom_strength))
        return None

    # Function to load and preprocess image
    def load_and_preprocess_image(image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0  # Normalize pixel values
        return image

    # Check if the numpy files exist
    if os.path.exists(class_weights_np_file) and os.path.exists(train_images_np_file) and \
            os.path.exists(train_labels_np_file) and os.path.exists(validation_images_np_file) and \
            os.path.exists(validation_labels_np_file):
        # Load the data from the numpy files
        print("Loading saved values..")
        train_images = np.load(train_images_np_file)
        train_labels = np.load(train_labels_np_file)
        validation_images = np.load(validation_images_np_file)
        validation_labels = np.load(validation_labels_np_file)
        class_weights = np.load(class_weights_np_file, allow_pickle=True).item()

    else:
        # Load the original annotation file
        annotation_file = "dataset/annotations/instances_Train.json"
        with open(annotation_file) as f:
            coco_data = json.load(f)

        images_info = coco_data["images"]
        categories = coco_data["categories"]
        annotations_info = coco_data["annotations"]

        image_paths = []
        labels = []

        # Process original dataset
        original_images_count = 0
        for annotation in annotations_info:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]

            category_name = next((category["name"] for category in categories if category["id"] == category_id), None)

            if category_name == "tree":
                attributes = annotation["attributes"]
                bloom_strength = attributes.get("bloom strenght")

                if bloom_strength is not None and bloom_strength != 0:  # Ignore images with bloom strength 0
                    image_path = next(
                        (os.path.join("dataset/images", image_info["file_name"]) for image_info in images_info if
                         image_info["id"] == image_id), None)

                    if image_path is not None:
                        image_paths.append(image_path)
                        labels.append(int(bloom_strength))
                        original_images_count += 1

        print(f"Loaded {original_images_count} images from original dataset")

        # Process new dataset
        new_images_count = 0
        ignored_images_count = 0

        # Load the new COCO annotations
        with open(new_annotation_file) as f:
            new_coco_data = json.load(f)

        new_annotations_info = new_coco_data["annotations"]
        new_images_info = new_coco_data["images"]

        # Create a dictionary to map image IDs to filenames
        image_id_to_filename = {img["id"]: img["file_name"] for img in new_images_info}

        for annotation in new_annotations_info:
            image_id = annotation["image_id"]
            filename = image_id_to_filename[image_id]

            bloom_strength = extract_bloom_strength(filename)

            if bloom_strength is not None:
                if bloom_strength != 0:  # Ignore images with bloom strength 0
                    image_path = os.path.join(new_image_dir, filename)
                    image_paths.append(image_path)
                    labels.append(bloom_strength)
                    new_images_count += 1
                else:
                    ignored_images_count += 1

        print(f"Loaded {new_images_count} new images from {new_image_dir}")
        print(f"Ignored {ignored_images_count} images with bloom strength 0")
        print(f"Total images now: {len(image_paths)}")

        # Load and preprocess all images
        images = [load_and_preprocess_image(image_path) for image_path in image_paths]
        images = np.array(images)
        labels = np.array(labels)

        print(f"Preprocessed {len(images)} images with corresponding labels")
        print(f"Unique labels before adjustment: {np.unique(labels)}")
        print(f"Label distribution before adjustment: {np.bincount(labels)[1:]}")  # Exclude count of label 0

        # Split the data into train and validation sets using stratified sampling
        train_images, validation_images, train_labels, validation_labels = train_test_split(images, labels,
                                                                                            test_size=0.2,
                                                                                            stratify=labels,
                                                                                            random_state=42)

        print(f"Training set: {len(train_images)} images")
        print(f"Validation set: {len(validation_images)} images")

        if model_type == 'classification':
            # Adjust labels to be in range 0-8 for neural network
            train_labels = train_labels - 1
            validation_labels = validation_labels - 1
            print(f"Unique labels after adjustment: {np.unique(train_labels)}")
            print(f"Label distribution after adjustment: {np.bincount(train_labels)}")

            # One-hot encode the labels
            from keras.utils import to_categorical
            train_labels = to_categorical(train_labels, num_classes=9)  # 9 classes (0-8)
            validation_labels = to_categorical(validation_labels, num_classes=9)
        else:  # regression or hierarchical_partial_labels
            # Keep labels as is (1-9 range)
            train_labels = train_labels.reshape(-1, 1)
            validation_labels = validation_labels.reshape(-1, 1)

        # Save the train and validation data as numpy files
        np.save(train_images_np_file, train_images)
        np.save(train_labels_np_file, train_labels)
        np.save(validation_images_np_file, validation_images)
        np.save(validation_labels_np_file, validation_labels)

        print("Saved preprocessed data to numpy files")

        # Calculate class weights
        if model_type == 'classification':
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes=np.unique(np.argmax(train_labels, axis=1)),
                                                              y=np.argmax(train_labels, axis=1))
            class_weights = dict(enumerate(class_weights))

            # Print out the class weights for each class and the number of images for that respective class
            for i in range(9):
                print(
                    f"Class {i} - Number of images: {np.sum(np.argmax(train_labels, axis=1) == i)}, Class weight: {class_weights[i]}")

            np.save(class_weights_np_file, class_weights)
        else:
            class_weights = None

    print("Data processing and saving completed.")

    # Print summary statistics
    print("Data Summary:")
    print(f"Total images: {len(image_paths)}")
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(validation_images)} images")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label distribution: {np.bincount(labels)[1:]}")  # Exclude count of label 0

    # Visualize sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(10):
        idx = np.random.randint(len(train_images))
        img = train_images[idx]
        label = np.argmax(train_labels[idx]) + 1 if model_type == 'classification' else train_labels[
            idx]  # Adjust label back to 1-9 range for display
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    return train_images, train_labels, validation_images, validation_labels, class_weights


if __name__ == '__main__':
    load_data(model_type='hierarchical_partial_labels')
