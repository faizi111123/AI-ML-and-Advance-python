import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np


# Check if dataset paths exist
if not os.path.exists('dataset/training_set'):
    print("Error: training_set directory not found")
    exit()
if not os.path.exists('dataset/test_set'):
    print("Error: test_set directory not found")
    exit()

# Set up GPU memory growth to avoid memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
    except:
        print("Memory growth setting failed")

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
print("Loading training data...")
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
print("Loading test data...")
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                           target_size=(64, 64),
                                           batch_size=32,
                                           class_mode='binary')

# Building the CNN with explicit Input layer
print("Building CNN model...")
inputs = tf.keras.layers.Input(shape=(64, 64, 3))
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

cnn = tf.keras.Model(inputs=inputs, outputs=outputs)
cnn.summary()

# Callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

# Compiling and Training
print("Compiling model...")
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training model...")
try:
    history = cnn.fit(
        training_set,
        validation_data=test_set,
        epochs=25,
        callbacks=[early_stopping]
    )
    
    # Save the model
    print("Saving model...")
    # Save as HDF5 format
    cnn.save('cat_dog_classifier.h5')
    print("Model saved as cat_dog_classifier.h5")
    
    # Also save the class indices for later use
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(training_set.class_indices, f)
    print("Class indices saved to class_indices.json")
    
except Exception as e:
    print(f"Training error: {e}")
    exit()

print("Training completed successfully!") 