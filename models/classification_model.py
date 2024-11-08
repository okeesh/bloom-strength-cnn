import os
from datetime import datetime

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from model_class.model_config import LearningRateScheduler

from model_class.model_config import ModelConfig, BaseModel
from model_class.utils import balanced_accuracy, BalancedDataGenerator


class ClassificationModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self):
        for layer in self.config.pretrained_model.layers:
            layer.trainable = False

        x = self.config.pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.num_classes, activation='softmax')(x)

        model = Model(inputs=self.config.pretrained_model.input, outputs=x)
        return model

    def compile_model(self, model,class_weights=None):
        # Calculate alpha values based on inverse class frequencies
        class_samples = np.array([117, 280, 99, 81, 84, 108, 308, 290, 50])
        total_samples = np.sum(class_samples)
        alpha = total_samples / (9 * class_samples)
        alpha = alpha / np.sum(alpha)  # Normalize to sum to 1
        print(alpha)

        # Create the loss function
        focal_loss = CategoricalFocalLoss(
            alpha=alpha,  # This is now a vector of 9 values
            gamma=7.0,
            from_logits=False,  # Set to True if your model doesn't include a final softmax layer
            label_smoothing=0
        )
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      #loss=focal_loss,
                        loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_data, validation_data, class_weights=None):
        experiment_directory = self.create_experiment_directory()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        lr_scheduler = LearningRateScheduler(new_lr=self.config.learning_rate * 0.1, change_epoch=20)

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        # if len(train_data) > 2:
        #     print("Using class weights")

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        #
        # bgen = BalancedDataGenerator(train_data[0], train_data[1], datagen, batch_size=16)
        # steps_per_epoch = bgen.steps_per_epoch
        #
        # # Get a single batch of data
        # x_batch, y_batch = bgen.__getitem__(0)

        # print("Shape of a single batch:")
        # print("x_batch shape:", x_batch.shape)
        # print("y_batch shape:", y_batch.shape)
        #
        # print("\nSample of y_batch (first 5 rows):")
        # print(y_batch[:5])
        #
        # print("\nClass distribution in this batch:")
        # class_counts = np.sum(y_batch, axis=0)
        # for i, count in enumerate(class_counts):
        #     print(f"Class {i}: {count}")

        # history = model.fit(
        #     bgen,
        #     steps_per_epoch=steps_per_epoch,
        #     validation_data=validation_data,
        #     epochs=self.config.epochs,
        #     verbose=1,
        #     callbacks=[lr_scheduler]
        # )

        history = model.fit(
            datagen.flow(train_data[0], train_data[1], batch_size=self.config.batch_size),
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=1,
            callbacks=[lr_scheduler]
        )


        # Give a final print for the training
        print(f"Final training accuracy: {history.history['accuracy'][-1]}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")

        # Print the early_stopping result if early stopped
        if early_stopping.stopped_epoch > 0:
            print(f"Early stopped at epoch {early_stopping.stopped_epoch} with validation loss {early_stopping.best}")

        # Plot confusion matrix
        y_pred = np.argmax(model.predict(validation_data[0]), axis=1)
        y_true = np.argmax(validation_data[1], axis=1)
        classes = [f'Class {i}' for i in range(self.config.num_classes)]

        # Calculate balanced accuracy
        bal_acc = balanced_accuracy(y_true, y_pred)
        print(f"Balanced Accuracy: {bal_acc:.4f}")

        # Add balanced accuracy to the history object
        history.history['balanced_accuracy'] = [bal_acc]

        # Add balanced accuracy to the history object
        history.history['balanced_accuracy'] = [bal_acc]

        predictions = model.predict(validation_data[0])
        true_labels = validation_data[1]

        #y_gen = [bgen.__getitem__(0)[1] for i in range(steps_per_epoch)]
        #print(np.unique(y_gen, return_counts=True))

        self.save_results(model, history, predictions, true_labels, experiment_directory)
        self.log_model_info(model, history, experiment_directory)
        self.plot_confusion_matrix(y_true, y_pred, classes, f'{experiment_directory}/classification_confusion_matrix.png')

        return history

    def create_experiment_directory(self):
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
        experiment_directory = f"experiments/classification/classification_{self.config.pretrained_model.name}_{timestamp}"
        os.makedirs(experiment_directory, exist_ok=True)
        return experiment_directory

    def log_model_info(self, model, history, experiment_directory):
        with open(f"{experiment_directory}/model_info.txt", "w") as f:
            f.write("Classification Model Information\n")
            f.write("=================================\n\n")

            f.write("Hyperparameters:\n")
            f.write(f"Learning rate: {self.config.learning_rate}\n")
            f.write(f"Dropout rate: {self.config.dropout_rate}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
            f.write(f"Epochs: {self.config.epochs}\n")
            f.write(f"Activation: {self.config.activation}\n")
            f.write(f"Number of classes: {self.config.num_classes}\n\n")

            f.write("Model Architecture:\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n")

            f.write("Training Performance:\n")
            f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"Balanced Accuracy: {history.history['balanced_accuracy'][0]:.4f}\n")

            # Trainingsgenauigkeit plotten
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Trainingsgenauigkeit')
            plt.ylabel('Genauigkeit')
            plt.xlabel('Epoche')
            plt.legend(['Training', 'Validierung'], loc='upper left')
            plt.savefig(f"{experiment_directory}/trainingsgenauigkeit.png")
            plt.show()

            # Trainingsverlust plotten
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Trainingsverlust')
            plt.ylabel('Verlust')
            plt.xlabel('Epoche')
            plt.legend(['Training', 'Validierung'], loc='upper left')
            plt.savefig(f"{experiment_directory}/trainingsverlust.png")
            plt.show()

import tensorflow as tf
from keras.losses import Loss


class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=None, gamma=2.0, from_logits=False, label_smoothing=0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.cast(alpha, dtype=tf.float32) if alpha is not None else None
        self.gamma = float(gamma)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)

        # Apply label smoothing if specified
        y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(tf.shape(y_true)[-1], dtype=tf.float32))

        # If predictions are logits, apply softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate focal loss
        focal_loss = tf.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)

        # Apply alpha weighting
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        # Sum over classes and samples
        loss = -tf.reduce_sum(y_true * focal_loss, axis=-1)

        return loss
