from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# load model
model = tf.keras.models.load_model("fruit_model.keras")

# your class names
class_names = [
    'apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit',
    'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit',
    'strawberry fruit'
]

def preprocess(image):
    image = image.resize((100, 100))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return "Fruit API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(file)

    processed = preprocess(image)
    prediction = model.predict(processed)

    return jsonify({
        "fruit": class_names[np.argmax(prediction)],
        "confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
