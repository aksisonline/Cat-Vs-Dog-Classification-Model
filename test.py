import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Function to predict an image
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Dog"
    else:
        return "Cat"

# Example usage
image_path = "testcat.jpg"  # Path to the image you want to predict
prediction = predict_image(image_path)
print("Prediction:", prediction)