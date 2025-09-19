import tensorflow as tf
import os


from tensorflow import keras
from keras.models import Sequential
from keras import layers

def save_model_for_deployment(model, model_name='my_model.h5'):
    """
    Save your trained model for deployment
    
    Args:
        model: Your trained Keras model
        model_name: Name for the saved model file
    """
    try:
        # Save the full model
        model.save(model_name)
        print(f"Model saved successfully as {model_name}")
        
        # Also save model architecture and weights separately (backup)
        model.save_weights(f"{model_name.replace('.h5', '_weights.h5')}")
        
        with open(f"{model_name.replace('.h5', '_architecture.json')}", 'w') as f:
            f.write(model.to_json())
        
        print("Model architecture and weights saved separately as backup")
        
        # Save class names
        
        class_names = [
            'Black Scurf', 
            'Blackleg', 
            'Common Scab', 
            'Dry Rot', 
            'Healthy', 
            'Miscellaneous', 
            'Pink Rot'
        ]
        
        import pickle
        with open('class_names.pkl', 'wb') as f:
            pickle.dump(class_names, f)
        
        print("Class names saved")
        
        return True
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
    
def create_sample_model():
    """
    Create a sample model architecture matching your training setup
    This is just for demonstration - replace with your actual trained model
    """
    
    model = Sequential([
        # Data augmentation layer
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        
        # Rescaling layer (if not done in preprocessing)
        layers.Rescaling(1./255),
        
        # Convolutional base
        layers.Conv2D(32, 3, activation='relu', input_shape=(255, 255, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(),
        
        # Classification head
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')  # 7 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
def test_model_loading():
    """Test if model can be loaded successfully"""
    model_path = 'my_model.h5'
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded successfully!")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            print(f"Total parameters: {model.count_params():,}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    else:
        print(f"✗ Model file not found: {model_path}")
        return False    
    
def convert_notebook_model():
    """
    Helper function to convert your notebook model to deployment format
    Add your model loading code from the notebook here
    """
    print("To convert your notebook model:")
    print("1. In your notebook, after training, add this code:")
    print()
    print("# Save the model for deployment")
    print("model.save('potato_disease_model.h5')")
    print()
    print("2. Then copy the .h5 file to your Flask app directory")
    print("3. Run this script to test loading: python model_utils.py")
if __name__ == "__main__":
    print("=== Potato Disease Model Utilities ===")
    print()
    
    # Test model loading
    print("Testing model loading...")
    success = test_model_loading()
    
    if not success:
        print()
        print("Model not found or couldn't be loaded.")
        convert_notebook_model()
    
print()
print("=== Model Setup Complete ===")