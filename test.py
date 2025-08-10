import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def predict_image(image_path, model_path='cat_dog_classifier.h5', indices_path='class_indices.json'):
    # Check if files exist
    if not os.path.exists(model_path):
        return f"Error: Model file not found at {model_path}"
    
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        if os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                class_indices = json.load(f)
            print(f"Class indices loaded: {class_indices}")
        else:
            # Default for cat-dog classifier (in case indices file is missing)
            class_indices = {'cat': 0, 'dog': 1}
            print(f"Class indices file not found, using default: {class_indices}")
        
        # Preprocess the image
        print(f"Loading test image from {image_path}...")
        test_image = load_img(image_path, target_size=(64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize the image
        
        # Make prediction
        print("Making prediction...")
        result = model.predict(test_image, verbose=0)
        
        # Determine class label
        if result[0][0] > 0.5:
            prediction = 'dog'
        else:
            prediction = 'cat'
        
        return f"Prediction: {prediction} (confidence: {result[0][0]:.2f})"
    
    except Exception as e:
        return f"Prediction error: {str(e)}"

if __name__ == "__main__":
    image_path = 'dataset/single_prediction/cat.jpeg'     
    result = predict_image(image_path)
    print(result) 