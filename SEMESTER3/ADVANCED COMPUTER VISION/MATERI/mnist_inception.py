import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST and preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Resize to 75x75 and normalize
x_train = tf.image.resize(tf.expand_dims(x_train, -1), [75, 75]) / 255.0
x_test = tf.image.resize(tf.expand_dims(x_test, -1), [75, 75]) / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define a simple Inception module
def inception_module(x, f1, f3r, f3, f5r, f5, pool_proj):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)
    
    # 1x1 conv -> 3x3 conv
    conv3 = layers.Conv2D(f3r, (1,1), padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(f3, (3,3), padding='same', activation='relu')(conv3)
    
    # 1x1 conv -> 5x5 conv
    conv5 = layers.Conv2D(f5r, (1,1), padding='same', activation='relu')(x)
    conv5 = layers.Conv2D(f5, (5,5), padding='same', activation='relu')(conv5)
    
    # 3x3 maxpool -> 1x1 conv
    pool = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool = layers.Conv2D(pool_proj, (1,1), padding='same', activation='relu')(pool)
    
    # Concatenate all filters
    output = layers.Concatenate()([conv1, conv3, conv5, pool])
    return output

# Build the model
inputs = tf.keras.Input(shape=(75, 75, 1))

x = layers.Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(inputs)
x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
x = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
x = layers.Conv2D(192, (3,3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

# Apply Inception modules
x = inception_module(x, 64, 96, 128, 16, 32, 32)
x = inception_module(x, 128, 128, 192, 32, 96, 64)
x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

# Flatten and Dense layers
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Compile and train
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

