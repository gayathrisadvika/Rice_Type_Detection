from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model("rice.h5")

# Ensure class labels are defined
class_labels = ["arborio", "basmati", "Ipsala", "jasmine", "Karacadag"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load and preprocess the image
    image = load_img(filepath, target_size=(128, 128))  # Resize to match model input
    image_array = img_to_array(image)  # Convert image to array
    image_array = image_array / 255.0  # Normalize (if needed)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return render_template("result.html", rice_type=predicted_class, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
