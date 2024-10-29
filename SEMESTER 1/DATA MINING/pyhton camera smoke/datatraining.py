import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Definisikan path untuk dataset
base_dir = 'I:/data mining sidang/datashet'
train_dir = 'I:/data mining sidang/datashet/training_data'
validation_dir = 'I:/data mining sidang/datashet/validation_data'

# Augmentasi gambar untuk data training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data validation hanya di-rescale
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Direktori training
    target_size=(150, 150),  # Resolusi gambar yang diharapkan
    batch_size=20,
    class_mode='binary'  # Kategori biner (merokok atau tidak merokok)
)

# Flow validation images in batches of 20 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # Direktori validation
    target_size=(150, 150),  # Resolusi gambar yang diharapkan
    batch_size=20,
    class_mode='binary'  # Kategori biner (merokok atau tidak merokok)
)

# Membangun Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model dengan optimizer Adam dan learning rate yang benar
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),  # Gunakan learning_rate bukan lr
    metrics=['accuracy']
)

# Melatih Model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Jumlah batch yang akan dieksekusi pada setiap epoch
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,  # Jumlah batch yang akan dieksekusi pada setiap epoch validasi
    verbose=2  # Tampilkan detail progress training
)

# Simpan model setelah pelatihan
model.save('model_merokok.h5')

# Evaluasi Model
acc = model.evaluate(validation_generator, verbose=2)

print(f'Validation accuracy: {acc[1]}')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
