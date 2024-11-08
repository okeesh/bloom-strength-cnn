import json
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend import epsilon, constant, ones
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from tensorflow import clip_by_value, reduce_sum
from tensorflow.python.framework.indexed_slices import math_ops
from tensorflow.python.framework.tensor_conversion_registry import constant_op
from tensorflow.python.ops import clip_ops
from model_class.model_config import BaseModel, ModelConfig
from tensorflow import math
from model_class.model_config import LearningRateScheduler
from model_class.utils import balanced_accuracy, BalancedDataGenerator, BalancedDataGeneratorMultiHot


class LearningRateTracker(Callback):
    def __init__(self):
        super().__init__()
        self.lr_changes = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.lr)
        if len(self.lr_changes) == 0 or lr != self.lr_changes[-1][1]:
            self.lr_changes.append((epoch + 1, lr))
class HierarchicalPartialLossModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self):
        # Zuerst alle Schichten einfrieren
        for layer in self.config.pretrained_model.layers:
            layer.trainable = False

        x = self.config.pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.dense_units, activation='relu', kernel_regularizer=l2(self.config.regularization_rate))(x)
        x = Dense(self.config.num_classes, activation="softmax")(x)

        model = Model(inputs=self.config.pretrained_model.input, outputs=x)
        return model
    def compile_model(self, model, class_weights=None):
        print(class_weights)
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss=HierarchicalPartialLossModel.CumulatedCrossEntropy,
                      metrics=[HierarchicalPartialLossModel.CumulatedAccuracy])
        return model

    def train_model(self, model, train_data, validation_data, class_weights=None):
        experiment_directory = self.create_experiment_directory()

        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        #lr_scheduler = LearningRateScheduler(new_lr=self.config.learning_rate * 0.1, change_epoch=20)
        lr_tracker = LearningRateTracker()

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        #
        # bgen = BalancedDataGeneratorMultiHot(train_data[0], train_data[1], datagen, batch_size=16)
        # steps_per_epoch = bgen.steps_per_epoch


        history = model.fit(
            datagen.flow(train_data[0], train_data[1], batch_size=self.config.batch_size),
            validation_data=validation_data,
            epochs=self.config.epochs,
            verbose=1,
            callbacks=[reduce_lr, lr_tracker],
            #class_weight=train_data[2] if len(train_data) > 2 else None
            #callbacks=[reduce_lr],

        )

        # history = model.fit(
        #     bgen,
        #     steps_per_epoch=steps_per_epoch,
        #     validation_data=validation_data,
        #     epochs=self.config.epochs,
        #     verbose=1,
        #     callbacks=[reduce_lr, lr_tracker],
        # )

        print(f"Final training custom metric: {history.history['CumulatedAccuracy'][-1]}")
        print(f"Final validation custom metric: {history.history['val_CumulatedAccuracy'][-1]}")

        # Ausgabe der Lernratenänderungen
        print("Lernratenänderungen:")
        for epoch, lr in lr_tracker.lr_changes:
            print(f"Epoche {epoch}: Lernrate geändert auf {lr}")


        # After training
        y_pred = np.argmax(model.predict(validation_data[0]), axis=1)
        y_true = np.argmax(validation_data[1], axis=1)

        # Calculate balanced accuracy
        bal_acc = balanced_accuracy(y_true, y_pred)
        print(f"Balanced Accuracy: {bal_acc:.4f}")

        # Add balanced accuracy to the history object
        history.history['balanced_accuracy'] = [bal_acc]


        classes = [f'Class {i}' for i in range(self.config.num_classes)]
        predictions = model.predict(validation_data[0])
        true_labels = validation_data[1]
        self.plot_confusion_matrix(y_true, y_pred, classes, f'{experiment_directory}/hierarchical_confusion_matrix.png')
        self.log_model_info(model, history, lr_tracker.lr_changes, experiment_directory)
        self.save_results(model, history, predictions, true_labels, lr_tracker.lr_changes, experiment_directory)
        return history

    def save_results(self, model, history, predictions, true_labels, lr_changes,  experiment_directory):
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, tf.Tensor):
                return obj.numpy().tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj

        results = {
            'model_name': self.config.pretrained_model.name,
            'model_type': self.config.model_type,
            'hyperparameters': {
                'learning_rate': self.config.learning_rate,
                'regularization_rate': self.config.regularization_rate,
                'dropout_rate': self.config.dropout_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'activation': self.config.activation,
                'dense_units': self.config.dense_units,
                'num_classes': self.config.num_classes,
            },
            'history': convert_to_serializable(history.history),
            'predictions': convert_to_serializable(predictions),
            'true_labels': convert_to_serializable(true_labels),
            'lr_changes': convert_to_serializable(lr_changes),
        }

        # Save as JSON
        with open(f"{experiment_directory}/results.json", 'w') as f:
            json.dump(results, f, indent=4)

        # Save predictions and true labels as pickle for easier loading later
        with open(f"{experiment_directory}/predictions_labels.pkl", 'wb') as f:
            pickle.dump({'predictions': predictions, 'true_labels': true_labels}, f)

        # Save model if needed
        model.save(f"{experiment_directory}/model.h5")

        print(f"Results saved in {experiment_directory}")

    def create_experiment_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_directory = f"experiments/hierarchical/hierarchical_{self.config.pretrained_model.name}_{timestamp}"
        os.makedirs(experiment_directory, exist_ok=True)
        return experiment_directory

    def log_model_info(self, model, history, lr_changes, experiment_directory):
        with open(f"{experiment_directory}/model_info.txt", "w") as f:
            f.write("Hierarchical Partial Loss Model Information\n")
            f.write("==========================================\n\n")

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

            f.write(f"Final training custom metric: {history.history['CumulatedAccuracy'][-1]:.4f}\n")
            f.write(f"Final validation custom metric: {history.history['val_CumulatedAccuracy'][-1]:.4f}\n")

            f.write("\nLearning Rate Changes:\n")
            for epoch, lr in lr_changes:
                f.write(f"Epoch {epoch}: Learning rate changed to {lr}\n")

        # # Plot training accuracy
        # plt.figure(figsize=(12, 6))
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('Modell Genauigkeit')
        # plt.ylabel('Genauigkeit')
        # plt.xlabel('Epoche')
        # plt.legend(['Training', 'Validierung'], loc='upper left')
        # plt.savefig(f"{experiment_directory}/accuracy.png")
        # plt.close()
        #
        # Plot training loss
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Modell Verlust')
        plt.ylabel('Verlust')
        plt.xlabel('Epoche')
        plt.legend(['Training', 'Validierung'], loc='upper left')
        plt.savefig(f"{experiment_directory}/loss.png")
        plt.close()

        # Plot custom metric
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['CumulatedAccuracy'])
        plt.plot(history.history['val_CumulatedAccuracy'])
        plt.title('Cumulated Accuracy')
        plt.ylabel('Cumulated Accuracy')
        plt.xlabel('Epoche')
        plt.legend(['Training', 'Validierung'], loc='upper left')
        plt.savefig(f"{experiment_directory}/cumulated_accuracy.png")
        plt.close()

    @staticmethod
    def create_hierarchical_labels(target):
        # Ensure input is a TensorFlow tensor
        if not isinstance(target, tf.Tensor):
            target = tf.convert_to_tensor(target)

        # Create a copy of the input tensor to avoid modifying the original
        hierarchical = tf.identity(target)

        # Replace all 1s with 2s
        hierarchical = tf.where(tf.equal(hierarchical, 1), tf.constant(2, dtype=hierarchical.dtype), hierarchical)

        # Find positions of all 2s
        positions = tf.where(tf.equal(hierarchical, 2))

        # Add 1s next to 2s
        for pos in positions:
            pos1_int64 = tf.cast(pos[1], tf.int64)  # Cast to int64
            if pos1_int64 > 0:
                hierarchical = tf.tensor_scatter_nd_update(
                    hierarchical,
                    [[pos[0], pos1_int64 - 1]],
                    [tf.maximum(hierarchical[pos[0], pos1_int64 - 1], 1.0)]
                )
            if pos1_int64 < tf.shape(hierarchical, out_type=tf.int64)[1] - 1:  # Ensure shape is also int64
                hierarchical = tf.tensor_scatter_nd_update(
                    hierarchical,
                    [[pos[0], pos1_int64 + 1]],
                    [tf.maximum(hierarchical[pos[0], pos1_int64 + 1], 1.0)]
                )

        return hierarchical
    @staticmethod
    def CumulatedCrossEntropy(target, output, axis=-1):
        mask_wide = clip_by_value(target, 0, 1)
        mask_narrow = clip_by_value(target, 1, 2) - 1

        output = output/(math_ops.reduce_sum(output, axis, True) + epsilon())
        epsilon_ = constant_op.constant(epsilon(), dtype= output.dtype.base_dtype)

        return -math_ops.log(clip_ops.clip_by_value(math_ops.reduce_sum(mask_wide * output, axis), epsilon_, 1. - epsilon_)) + -math_ops.log(clip_ops.clip_by_value(math_ops.reduce_sum(mask_narrow * output, axis), epsilon_, 1. - epsilon_))
    # @staticmethod
    # def CumulatedAccuracy(y_true, y_pred, axis=-1):
    #     max_value = tf.reduce_max(y_true)
    #     mask_narrow = clip_by_value(y_true, max_value -1, max_value) - (max_value - 1)
    #     truths = math.argmax(mask_narrow * (y_pred + 1), axis)  #max aller werte eingeschränkt auf die erlaubten positionen
    #     preds = math.argmax(y_pred, axis) #max aller ausgaben
    #     marks = math.equal(truths, preds) #ist max aller ausgaben auf einer erlaubten position?
    #     accuracy = math_ops.reduce_mean(
    #         tf.cast(marks, tf.float32))  # Der Prozentsatz der als zulässig klassifizierten Labels
    #     return accuracy
    @staticmethod
    def CumulatedAccuracy(y_true, y_pred, axis=-1):
        # Create a "wide mask" to include both '2' and neighboring '1's
        mask_wide = tf.cast(y_true > 0, tf.float32)  # This turns all values > 0 to '1'

        # Restrict predictions to valid positions by applying the wide mask
        truths = tf.argmax(mask_wide * (y_pred + 1), axis)  # Find max values in valid positions
        preds = tf.argmax(y_pred, axis)  # Model's predicted class (highest logit or probability)

        # Check if the prediction is in a valid hierarchical position
        marks = tf.equal(truths, preds)

        # Calculate accuracy: percentage of correctly predicted valid positions
        accuracy = tf.reduce_mean(tf.cast(marks, tf.float32))
        return accuracy
