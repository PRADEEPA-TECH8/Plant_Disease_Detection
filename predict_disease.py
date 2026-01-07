import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pyttsx3
from Solutions.Disease_Solutions import disease_solutions

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def predict_plant_disease(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map index to class name using disease_solutions dictionary keys
    class_names = list(disease_solutions.keys())
    predicted_class_key = class_names[predicted_class]
    predicted_class_name = predicted_class_key.replace('_', ' ')

    print("Predicted Class Key:", predicted_class_key)  # Debugging key
    print("Predicted Disease:", predicted_class_name)

    # Get suggested solution
    solution = disease_solutions.get(predicted_class_key, "Solution not found for this disease.")
    print("Suggested Solution:", solution)

    # Speak both together
    full_message = f"The predicted disease is {predicted_class_name}. The suggested solution is: {solution}"
    speak(full_message)

# Test the function
predict_plant_disease("test_leaf.jpg")

