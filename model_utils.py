import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import os
import time

class PotatoDiseasePredictor:
    """Class to handle model loading and predictions"""
    
    def __init__(self, model_path="my_model.h5"):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Black Scurf', 
            'Blackleg', 
            'Common Scab', 
            'Dry Rot', 
            'Healthy', 
            'Miscellaneous', 
            'Pink Rot'
        ]
        self.img_size = (255, 255)
        
    def load_model(self):
        """Load the trained Keras model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                print(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to model input size
            img = img.resize(self.img_size)
            
            # Convert to numpy array
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction on image"""
        if self.model is None:
            return None, 0.0, []
        
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None, 0.0, []
        
        try:
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Sort predictions with confidence
            all_predictions = [
                {'class': self.class_names[i], 'confidence': float(predictions[0][i])}
                for i in range(len(self.class_names))
            ]
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return predicted_class, confidence, all_predictions
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, 0.0, []
    
    def validate_image(self, image_path):
        """Validate if image is suitable for prediction"""
        try:
            img = Image.open(image_path)
            if img.size[0] < 50 or img.size[1] < 50:
                return False, "Image is too small."
            if img.mode not in ['RGB', 'RGBA', 'L']:
                return False, "Unsupported image format."
            return True, "Image is valid"
        except Exception as e:
            return False, f"Error validating image: {e}"


def create_upload_folder():
    """Create upload folder if it doesn't exist"""
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return upload_folder


def cleanup_old_files(folder_path, max_age_hours=24):
    """Clean up old uploaded files"""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Error cleaning up {filename}: {e}")


def get_model_info(model_path='potato_disease_model.h5'):
    """Get information about the loaded model"""
    try:
        if not os.path.exists(model_path):
            return None
        model = load_model(model_path)
        return {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_params': model.count_params(),
            'layers': len(model.layers)
        }
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None


if __name__ == "__main__":
    predictor = PotatoDiseasePredictor()
    
    if predictor.load_model():
        print("Model loaded successfully!")
        print(f"Available classes: {predictor.class_names}")
        
        # Allow user to test predictions directly
        while True:
            image_path = input("\nEnter image path (or 'q' to quit): ").strip()
            if image_path.lower() == 'q':
                break
            
            if not os.path.exists(image_path):
                print("Image not found. Try again.")
                continue
            
            valid, msg = predictor.validate_image(image_path)
            if not valid:
                print(f"Invalid image: {msg}")
                continue
            
            predicted_class, confidence, all_predictions = predictor.predict(image_path)
            if predicted_class:
                print(f"\nPredicted Class: {predicted_class}")
                print(f"Confidence: {confidence:.2f}")
                print("\nAll Class Probabilities:")
                for p in all_predictions:
                    print(f"- {p['class']}: {p['confidence']:.4f}")
            else:
                print("Prediction failed.")
    else:
        print("Failed to load model.")