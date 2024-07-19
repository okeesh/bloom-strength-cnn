import math
from keras.applications import ResNet50, MobileNetV2

from keras.backend import epsilon
from keras.regularizers import l2
from tensorflow import clip_by_value
from tensorflow.python.framework.indexed_slices import math_ops
from tensorflow.python.framework.tensor_conversion_registry import constant_op
from tensorflow.python.ops import clip_ops

import tensorflow as tf

from load_data import load_data, delete_saved_data


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


def CumulatedCrossEntropy(output, target, axis=-1):
    target_morphed = create_hierarchical_labels(target)
    mask_wide = clip_by_value(target_morphed, 0, 1)
    mask_narrow = clip_by_value(target_morphed, 1, 2) - 1

    # Konvertiere die Ausgabe in float32, um sicherzustellen, dass sie den gleichen Datentyp wie epsilon hat
    output = tf.cast(output, tf.float32)

    # Berechne die Summe der Ausgabe entlang der angegebenen Achse
    output_sum = math_ops.reduce_sum(output, axis, keepdims=True)

    # Addiere epsilon zur Ausgabensumme, um Division durch Null zu vermeiden
    output_sum_epsilon = output_sum + tf.keras.backend.epsilon()

    # Normalisiere die Ausgabe durch Division durch die Ausgabensumme mit Epsilon
    output_normalized = output / output_sum_epsilon

    epsilon_ = tf.keras.backend.epsilon()
    return -math_ops.log(clip_ops.clip_by_value(math_ops.reduce_sum(mask_wide * output_normalized, axis), epsilon_,
                                                1. - epsilon_)) + -math_ops.log(
        clip_ops.clip_by_value(math_ops.reduce_sum(mask_narrow * output_normalized, axis), epsilon_, 1. - epsilon_))


def CumulatedAccuracy(y_true, y_pred, axis=-1):
    y_true_argmax = tf.argmax(y_true, axis)
    y_pred_argmax = tf.argmax(y_pred, axis)

    # Convert to float for subtraction
    y_true_float = tf.cast(y_true_argmax, tf.float32)
    y_pred_float = tf.cast(y_pred_argmax, tf.float32)

    # Calculate the absolute difference
    diff = tf.abs(y_true_float - y_pred_float)

    # Consider predictions correct if they're exact or off by one level
    correct_predictions = tf.less_equal(diff, 1.0)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam

def create_model(num_classes):
    # Load pretrained ResNet50 Model
    base_model = MobileNetV2(weights='imagenet', include_top=False)

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(9, activation='softmax')(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile model with custom loss function
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=CumulatedCrossEntropy,
                  metrics=['accuracy', CumulatedAccuracy])

    return model

def train_model():

   # delete_saved_data()
    # Load the data
    train_images, train_labels, validation_images, validation_labels, class_weights = load_data("hierarchical")

    # Create the model
    model = create_model(9)  # 9 classes

    # Train the model
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=50,  # Number of epochs
        batch_size=16,  # Batch size
        verbose=1
    )

    # Save the trained model
    model.save('trained_model.h5')

    return history

# Call the train_model function
history = train_model()