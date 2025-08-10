import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# CONFIG
DATA_DIR = "spoken_numbers_pcm"
SAMPLE_RATE = 16000
DURATION = 1  # sec
N_MFCC = 40
MAX_PAD_LEN = 40
NUM_CLASSES = 10
EPOCHS = 30
BATCH_SIZE = 32

# Extract MFCC features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # Shape: (time, n_mfcc)
    if mfcc.shape[0] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_PAD_LEN]
    return mfcc

# Load dataset
def load_dataset(data_dir):
    X, y, labels = [], [], []
    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path): continue
        labels.append(label)
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                path = os.path.join(label_path, file)
                mfcc = extract_features(path)
                X.append(mfcc)
                y.append(label)
    label_to_index = {l: i for i, l in enumerate(sorted(set(y)))}
    y_idx = [label_to_index[label] for label in y]
    return np.array(X), tf.keras.utils.to_categorical(y_idx), label_to_index

# Load data
X, y, label_to_index = load_dataset(DATA_DIR)
index_to_label = {v: k for k, v in label_to_index.items()}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Model
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((MAX_PAD_LEN, N_MFCC), NUM_CLASSES)
model.summary()

# Train
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
]

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot training
def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.show()

plot_history(history)

# Predict
def predict_sample(sample, model, index_to_label):
    pred = model.predict(np.expand_dims(sample, axis=0))
    return index_to_label[np.argmax(pred)]

# Try 5 test samples
for i in range(5):
    true_label = index_to_label[np.argmax(y_test[i])]
    pred_label = predict_sample(X_test[i], model, index_to_label)
    print(f"True: {true_label}, Predicted: {pred_label}")

# Predict custom file
def recognize_file(file_path):
    mfcc = extract_features(file_path)
    pred_label = predict_sample(mfcc, model, index_to_label)
    print(f"Recognized speech from '{file_path}': {pred_label}")

# Example usage
recognize_file("spoken_numbers_pcm/two/2_Agnes_100.wav")
