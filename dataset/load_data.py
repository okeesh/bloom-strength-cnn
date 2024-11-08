import json
import logging
import os
import re
import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from collections import Counter
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
    # def process_annotation_file(annotation_file, image_dir):
    #     with open(annotation_file) as f:
    #         coco_data = json.load(f)
    #
    #     categories = {cat['id']: float(cat['name']) for cat in coco_data['categories'] if cat['name'] != 'tree'}
    #     annotations = coco_data['annotations']
    #     images = coco_data['images']
    #
    #     image_paths = []
    #     labels = []
    #     bboxes = []
    #
    #     for annotation in annotations:
    #         image_id = annotation['image_id']
    #         category_id = annotation['category_id']
    #         bbox = annotation['bbox']
    #
    #         if category_id in categories:
    #             bloom_strength = categories[category_id]
    #             image_info = next((img for img in images if img['id'] == image_id), None)
    #
    #             if image_info:
    #                 image_path = os.path.join(image_dir, image_info['file_name'])
    #                 image_paths.append(image_path)
    #                 labels.append(bloom_strength)
    #                 bboxes.append(bbox)
    #
    #     print(labels)
    #
    #     return image_paths, labels, bboxes

    global class_weights_dict, class_weights

    def process_annotation_file(annotation_file, image_dir, class_limit=100):
        try:
            with open(annotation_file) as f:
                coco_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading annotation file: {e}")
            return [], [], []

        categories = {cat['id']: float(cat['name']) for cat in coco_data['categories']
                      if cat['name'] != 'tree' and cat['name'].replace('.', '').isdigit()}

        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

        image_paths = []
        labels = []
        bboxes = []
        class_counter = Counter()

        for annotation in coco_data['annotations']:
            category_id = annotation['category_id']
            if category_id in categories:
                bloom_strength = categories[category_id]

                # # Check if we've reached the limit for this class
                # if class_counter[bloom_strength] >= class_limit:
                #     continue

                image_id = annotation['image_id']
                if image_id in image_id_to_filename:
                    image_path = os.path.join(image_dir, image_id_to_filename[image_id])
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        labels.append(bloom_strength)
                        bboxes.append(annotation['bbox'])
                        class_counter[bloom_strength] += 1
                    else:
                        logging.warning(f"Image file not found: {image_path}")
                else:
                    logging.warning(f"Image ID {image_id} not found in image list")

        print(f"Processed {len(image_paths)} valid annotations")
        print(f"Class distribution: {dict(class_counter)}")

        # Calculate and log the percentage of each class
        total_samples = sum(class_counter.values())
        class_percentages = {cls: (count / total_samples) * 100 for cls, count in class_counter.items()}
        print(f"Class percentages: {class_percentages}")

        return image_paths, labels, bboxes

    # Define the subdirectory for the numpy files
    subdirectory = os.path.join(SCRIPT_DIR, 'dataset', 'numpy_files')
    print(f"Numpy files directory: {subdirectory}")

    file_prefix = model_type

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
    class_weights_dict = None

    # Check if the numpy files exist
    if os.path.exists(train_images_np_file) and \
            os.path.exists(train_labels_np_file) and os.path.exists(validation_images_np_file) and \
            os.path.exists(validation_labels_np_file):
        # Load the data from the numpy files
        print("Loading saved values..")
        train_images = np.load(train_images_np_file)
        train_labels = np.load(train_labels_np_file)
        validation_images = np.load(validation_images_np_file)
        validation_labels = np.load(validation_labels_np_file)


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

        def get_class_distribution(labels, model_type):
            if model_type == 'regression':
                # For regression, we'll create bins
                bins = np.arange(1, 11)  # Assuming labels are from 1 to 9
                hist, _ = np.histogram(labels, bins=bins)
                class_counter = dict(zip(range(1, 10), hist))
            else:
                # For classification, labels are integers
                class_counter = Counter(labels.flatten())

            total_samples = len(labels)
            class_percentages = {cls: (count / total_samples) * 100
                                 for cls, count in class_counter.items()}
            return class_counter, class_percentages

        # Calculate and print class distribution for training set
        train_class_counter, train_class_percentages = get_class_distribution(train_labels, model_type)

        print("Training Set Class Distribution:")
        print(f"Raw counts: {dict(train_class_counter)}")
        print(f"Percentages: {train_class_percentages}")

        # Calculate and print class distribution for validation set
        val_class_counter, val_class_percentages = get_class_distribution(validation_labels, model_type)

        print("\nValidation Set Class Distribution:")
        print(f"Raw counts: {dict(val_class_counter)}")
        print(f"Percentages: {val_class_percentages}")
        class_weights_dict = None  # Initialize to None
        # Process labels based on model type
        if model_type == 'classification' or model_type == 'hierarchical':
            # Adjust labels and one-hot encode
            train_labels = train_labels - 1
            validation_labels = validation_labels - 1
            train_labels_categorical = to_categorical(train_labels, num_classes=9)
            validation_labels_categorical = to_categorical(validation_labels, num_classes=9)

            # Calculate class weights
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes=np.unique(train_labels),
                                                              y=train_labels)
            class_weights_dict = dict(enumerate(class_weights))
            print("Class weights:", class_weights_dict)

            if model_type == 'hierarchical':
                train_labels = HierarchicalPartialLossModel.create_hierarchical_labels(train_labels_categorical)
                validation_labels = HierarchicalPartialLossModel.create_hierarchical_labels(
                    validation_labels_categorical)
            else:  # classification
                train_labels = train_labels_categorical
                validation_labels = validation_labels_categorical
        else:  # regression
                print("Loading Regression Labels..")
                train_labels = train_labels.reshape(-1, 1)
                validation_labels = validation_labels.reshape(-1, 1)

        # Save processed data
        os.makedirs(subdirectory, exist_ok=True)
        np.save(train_images_np_file, train_images)
        np.save(train_labels_np_file, train_labels)
        np.save(validation_images_np_file, validation_images)
        np.save(validation_labels_np_file, validation_labels)
        if class_weights_dict is not None:
            np.save(class_weights_np_file, class_weights_dict)


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

    if os.path.exists(class_weights_np_file):
        class_weights_dict = np.load(class_weights_np_file, allow_pickle=True).item()
        print("Loaded class weights:", class_weights_dict)

    return train_images, train_labels, validation_images, validation_labels, class_weights_dict


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
    load_data(model_type='classification')