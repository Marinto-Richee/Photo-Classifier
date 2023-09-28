import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define data directories
data_dir = "images"

# Define data augmentation
datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset using tf.data.Dataset
train_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='binary',
    class_names=['not_accepted', 'accepted'],
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

val_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='binary',
    class_names=['not_accepted', 'accepted'],
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
num_epochs = 10
model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

# Save the model to a file
model.save('Photo_classifier_tf_V1.h5')
