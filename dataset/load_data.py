import json
import os
import re
import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def delete_saved_data(subdirectory='numpy_files'):
    # Define the paths for the numpy files
    subdirectory_path = os.path.join(SCRIPT_DIR, subdirectory)
    train_images_np_file = os.path.join(subdirectory_path, 'train_images.npy')
    train_labels_np_file = os.path.join(subdirectory_path, 'train_labels.npy')
    validation_images_np_file = os.path.join(subdirectory_path, 'validation_images.npy')
    validation_labels_np_file = os.path.join(subdirectory_path, 'validation_labels.npy')
    class_weights_np_file = os.path.join(subdirectory_path, 'class_weights.npy')

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
    subdirectory = os.path.join(SCRIPT_DIR, 'dataset', 'numpy_files')
    print(f"Numpy files directory: {subdirectory}")

    # Define the prefix for the numpy files based on the model type
    if model_type != 'regression':
        # works for classification and hierarchichal
        file_prefix = 'classification'
    else:
        # For 'regression', use the  prefix 'regression'
        file_prefix = 'regression'

    # Define the paths for the numpy files
    train_images_np_file = os.path.join(subdirectory, f'{file_prefix}_train_images.npy')
    train_labels_np_file = os.path.join(subdirectory, f'{file_prefix}_train_labels.npy')
    validation_images_np_file = os.path.join(subdirectory, f'{file_prefix}_validation_images.npy')
    validation_labels_np_file = os.path.join(subdirectory, f'{file_prefix}_validation_labels.npy')
    class_weights_np_file = os.path.join(subdirectory, f'{file_prefix}_class_weights.npy')

    # Initialize variables
    train_images = None
    train_labels = None
    validation_images = None
    validation_labels = None
    class_weights = None

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

    # If any of the required data is not loaded, process the dataset
    if train_images is None or train_labels is None or validation_images is None or validation_labels is None:
        # Define paths for the datasets
        new_image_dir = os.path.join(SCRIPT_DIR, "dataset", "images_title_lable")
        new_annotation_file = os.path.join(SCRIPT_DIR, "dataset", "annotations", "new_annotations.json")
        annotation_file = os.path.join(SCRIPT_DIR, "dataset", "annotations", "instances_Train.json")

        def extract_bloom_strength(filename):
            match = re.search(r'_(\d+\.\d+)_', filename)
            if match:
                bloom_strength = float(match.group(1))
                return int(round(bloom_strength))
            return None

        def load_and_preprocess_image(image_path, bbox=None):
            image = Image.open(image_path)
            if bbox:
                x, y, w, h = bbox
                image = image.crop((x, y, x + w, y + h))
            image = image.resize((224, 224))
            image = img_to_array(image)
            image = image / 255.0  # Normalize pixel values
            return image

        # Load the original annotation file
        with open(annotation_file) as f:
            coco_data = json.load(f)

        images_info = coco_data["images"]
        categories = coco_data["categories"]
        annotations_info = coco_data["annotations"]

        image_paths = []
        labels = []
        bboxes = []

        # Process original dataset
        original_images_count = 0
        for annotation in annotations_info:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            bbox = annotation["bbox"]

            category_name = next((category["name"] for category in categories if category["id"] == category_id), None)

            if category_name == "tree":
                attributes = annotation["attributes"]
                bloom_strength = attributes.get("bloom strenght")

                if bloom_strength is not None and bloom_strength != 0:  # Ignore images with bloom strength 0
                    image_path = next(
                        (os.path.join(SCRIPT_DIR, "dataset", "images", image_info["file_name"]) for image_info in
                         images_info if
                         image_info["id"] == image_id), None)

                    if image_path is not None:
                        image_paths.append(image_path)
                        labels.append(int(bloom_strength))
                        bboxes.append(bbox)
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

        # Create a dictionary to map image IDs to filenames and bounding boxes
        image_id_to_info = {img["id"]: {"file_name": img["file_name"], "bbox": None} for img in new_images_info}

        for annotation in new_annotations_info:
            image_id = annotation["image_id"]
            image_id_to_info[image_id]["bbox"] = annotation["bbox"]

        for image_id, info in image_id_to_info.items():
            filename = info["file_name"]
            bbox = info["bbox"]

            bloom_strength = extract_bloom_strength(filename)

            if bloom_strength is not None:
                if bloom_strength != 0:  # Ignore images with bloom strength 0
                    image_path = os.path.join(new_image_dir, filename)
                    image_paths.append(image_path)
                    labels.append(bloom_strength)
                    bboxes.append(bbox)  # Now we're adding the bounding box for the new dataset
                    new_images_count += 1
                else:
                    ignored_images_count += 1

        print(f"Loaded {new_images_count} new images from {new_image_dir}")
        print(f"Ignored {ignored_images_count} images with bloom strength 0")
        print(f"Total images now: {len(image_paths)}")


        # Load and preprocess all images
        images = [load_and_preprocess_image(image_path, bbox) for image_path, bbox in zip(image_paths, bboxes)]
        images = np.array(images)
        labels = np.array(labels)

        # Visualize sample images after bounding box application
        plt.figure(figsize=(15, 6))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            idx = np.random.randint(len(images))
            plt.imshow(images[idx])
            plt.title(f"Label: {labels[idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Split the data into train and validation sets
        train_images, validation_images, train_labels, validation_labels = train_test_split(
            images, labels, test_size=0.3, stratify=labels, random_state=42
        )

        # Process labels based on model type
        if file_prefix == 'classification':
            # Adjust labels and one-hot encode
            train_labels = train_labels - 1
            validation_labels = validation_labels - 1
            train_labels = to_categorical(train_labels, num_classes=9)
            print("Sample train label after one-hot encoding:", train_labels[0])
            validation_labels = to_categorical(validation_labels, num_classes=9)

            # Calculate class weights
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes=np.unique(np.argmax(train_labels, axis=1)),
                                                              y=np.argmax(train_labels, axis=1))
            class_weights = dict(enumerate(class_weights))
        else:  # regression
            train_labels = train_labels.reshape(-1, 1)
            validation_labels = validation_labels.reshape(-1, 1)
            class_weights = None

        # Save processed data
        os.makedirs(subdirectory, exist_ok=True)
        np.save(train_images_np_file, train_images)
        np.save(train_labels_np_file, train_labels)
        np.save(validation_images_np_file, validation_images)
        np.save(validation_labels_np_file, validation_labels)
        if class_weights is not None:
            np.save(class_weights_np_file, class_weights)

    print("Data processing and saving completed.")

    # Print summary statistics
    print("Data Summary:")
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(validation_images)} images")

    # Visualize sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(10):
        idx = np.random.randint(len(train_images))
        img = train_images[idx]
        label = np.argmax(train_labels[idx]) + 1 if model_type == 'classification' else train_labels[idx]
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    return train_images, train_labels, validation_images, validation_labels, class_weights


if __name__ == '__main__':
    load_data(model_type='hierarchical_partial_labels')