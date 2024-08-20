import json
import os
import numpy as np
from PIL import Image
from keras.utils import img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


def load_data(model_type='classification'):
    # Define the subdirectory for the numpy files
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subdirectory = os.path.join(SCRIPT_DIR, 'dataset', 'numpy_files')
    print(f"Numpy files directory: {subdirectory}")

    # Define the prefix for the numpy files based on the model type
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
    class_weights = None

    def should_load_saved_data(file_prefix):
        base_files_exist = (
                os.path.exists(train_images_np_file) and
                os.path.exists(train_labels_np_file) and
                os.path.exists(validation_images_np_file) and
                os.path.exists(validation_labels_np_file)
        )

        if not base_files_exist:
            return False

        if file_prefix == "classification":
            return os.path.exists(class_weights_np_file)
        else:
            return not os.path.exists(class_weights_np_file)

    # Check if the numpy files exist
    if should_load_saved_data(file_prefix):
        # Load the data from the numpy files
        print("Loading saved values..")
        train_images = np.load(train_images_np_file)
        train_labels = np.load(train_labels_np_file)
        validation_images = np.load(validation_images_np_file)
        validation_labels = np.load(validation_labels_np_file)
        if file_prefix == "classification":
            class_weights = np.load(class_weights_np_file, allow_pickle=True).item()
    else:
        # Define paths for the datasets
        image_dir = os.path.join(SCRIPT_DIR, "dataset", "allpics")
        annotation_file = os.path.join(SCRIPT_DIR, "dataset", "allpics", "annotations", "instances_default.json")

        print(f"Image directory: {image_dir}")
        print(f"Annotation file: {annotation_file}")

        def load_and_preprocess_image(image_path):
            try:
                image = Image.open(image_path)
                image = image.resize((224, 224))
                image = img_to_array(image)
                image = image / 255.0  # Normalize pixel values
                return image
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                return None

        # Load the COCO annotation file
        try:
            with open(annotation_file) as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"Error loading annotation file: {str(e)}")
            return None, None, None, None, None

        images_info = {img['id']: img for img in coco_data["images"]}
        categories = {cat['id']: cat['name'] for cat in coco_data["categories"]}

        image_paths = []
        labels = []

        # Process COCO dataset
        for image in coco_data["images"]:
            image_id = image["id"]
            file_name = image["file_name"]

            # Extract bloom strength from file name
            bloom_strength = int(file_name.split('_')[1].split('.')[0])

            if bloom_strength != 0:  # Ignore images with bloom strength 0
                image_path = os.path.join(image_dir, file_name)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    labels.append(bloom_strength)
                else:
                    print(f"Image file not found: {image_path}")

        print(f"Found {len(image_paths)} valid images from COCO dataset")

        if len(image_paths) == 0:
            print("No valid images found. Check your data and file paths.")
            return None, None, None, None, None

        # Load and preprocess all images
        images = [img for img in (load_and_preprocess_image(image_path) for image_path in image_paths) if
                  img is not None]

        if len(images) == 0:
            print("No images could be successfully processed. Check your image files and preprocessing function.")
            return None, None, None, None, None

        images = np.array(images)
        labels = np.array(labels[:len(images)])  # Ensure labels match the number of successfully processed images

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
            validation_labels = to_categorical(validation_labels, num_classes=9)

            # Calculate class weights
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes=np.unique(np.argmax(train_labels, axis=1)),
                                                              y=np.argmax(train_labels, axis=1))
            class_weights = dict(enumerate(class_weights))
        elif file_prefix == "regression":
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
    if len(train_images) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        for i in range(min(10, len(train_images))):
            idx = np.random.randint(len(train_images))
            img = train_images[idx]
            label = np.argmax(train_labels[idx]) + 1 if model_type == 'classification' else train_labels[idx]
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No images to visualize.")

    print(train_labels)

    return train_images, train_labels, validation_images, validation_labels, class_weights


if __name__ == '__main__':
    load_data(model_type='classification')