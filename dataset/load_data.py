import json
import os
import re
import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

from models.hierarchicalpartial_model import HierarchicalPartialLossModel

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
    def process_annotation_file(annotation_file, image_dir):
        with open(annotation_file) as f:
            coco_data = json.load(f)

        categories = {cat['id']: float(cat['name']) for cat in coco_data['categories'] if cat['name'] != 'tree'}
        annotations = coco_data['annotations']
        images = coco_data['images']

        image_paths = []
        labels = []
        bboxes = []

        for annotation in annotations:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            bbox = annotation['bbox']

            if category_id in categories:
                bloom_strength = categories[category_id]
                image_info = next((img for img in images if img['id'] == image_id), None)

                if image_info:
                    image_path = os.path.join(image_dir, image_info['file_name'])
                    image_paths.append(image_path)
                    labels.append(bloom_strength)
                    bboxes.append(bbox)

        print(labels)

        return image_paths, labels, bboxes

    # Define the subdirectory for the numpy files
    subdirectory = os.path.join(SCRIPT_DIR, 'dataset', 'numpy_files')
    print(f"Numpy files directory: {subdirectory}")

    # Define the prefix for the numpy files based on the model type
    if model_type != 'regression':
        # works for classification and hierarchical
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
        image_dir = os.path.join(SCRIPT_DIR, "dataset", "allpics")
        annotation_file = os.path.join(SCRIPT_DIR, "dataset", "allpics", "annotations", "instances_default.json")

        image_paths, labels, bboxes = process_annotation_file(annotation_file, image_dir)

        # Load and preprocess all images
        images = [load_and_preprocess_image(image_path, bbox) for image_path, bbox in zip(image_paths, bboxes)]
        images = [img for img in images if img is not None]  # Remove any None values
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
        if model_type == 'classification':
            # Adjust labels and one-hot encode
            train_labels = train_labels - 1
            validation_labels = validation_labels - 1
            train_labels = to_categorical(train_labels, num_classes=9)
            validation_labels = to_categorical(validation_labels, num_classes=9)
            print("Sample train label after one-hot encoding:", train_labels[0])


            # Calculate class weights
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes=np.unique(np.argmax(train_labels, axis=1)),
                                                              y=np.argmax(train_labels, axis=1))
            class_weights = dict(enumerate(class_weights))
        elif model_type == 'hierarchical':
            # Adjust labels and one-hot encode
            train_labels = train_labels - 1
            validation_labels = validation_labels - 1
            train_labels = to_categorical(train_labels, num_classes=9)
            validation_labels = to_categorical(validation_labels, num_classes=9)
            train_labels = HierarchicalPartialLossModel.create_hierarchical_labels(train_labels)
            validation_labels = HierarchicalPartialLossModel.create_hierarchical_labels(validation_labels)
            print("Sample train label after one-hot encoding:", train_labels[0])

        else:  # regression
                print("HIHIHIHIHI")
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


def load_and_preprocess_image(image_path, bbox=None):
    try:
        image = Image.open(image_path)
        if bbox:
            x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = image / 255.0  # Normalize pixel values
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


if __name__ == '__main__':
    load_data(model_type='hierarchical_partial_labels')