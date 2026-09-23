import os

import numpy as np
import keras
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "laptop_screen_damage_model.keras"
IMG_SIZE = (224, 224)

# Load model once when the application starts
model = keras.models.load_model(MODEL_PATH)


def predict_image(image):
    image = Image.open(image).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    if probability >= 0.5:
        prediction = "normal"
        confidence = probability
    else:
        prediction = "broken"
        confidence = 1 - probability

    return prediction, confidence


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select an image."
        )

    image = request.files["image"]

    if image.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:
        prediction, confidence = predict_image(image)

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=f"{confidence * 100:.2f}%"
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )