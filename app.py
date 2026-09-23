import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (224, 224)

MODEL_PATH = "best_screen_damage_model.keras"

# IMPORTANT:
# Replace these with the EXACT output of:
#
# print("Classes:", train_ds.class_names)
#
# Example:
# ['damaged', 'not_damaged']

CLASS_NAMES = ['broken', 'normal']


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Load trained TensorFlow model
# ============================================================

print("Loading TensorFlow model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Home page
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get uploaded file
        # ----------------------------------------------------

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            file
        ).convert("RGB")


        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        image = image.resize(
            IMG_SIZE
        )


        # ----------------------------------------------------
        # Convert image to NumPy
        # ----------------------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Add batch dimension
        # Shape:
        # (224, 224, 3)
        #       ↓
        # (1, 224, 224, 3)
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )
        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # Sigmoid output
        # ----------------------------------------------------

        probability = float(
            prediction[0][0]
        )


        # ----------------------------------------------------
        # Classification
        #
        # label 0 = CLASS_NAMES[0]
        # label 1 = CLASS_NAMES[1]
        # ----------------------------------------------------

        if probability >= 0.5:

            predicted_class = CLASS_NAMES[1]

            confidence = probability

        else:

            predicted_class = CLASS_NAMES[0]

            confidence = 1 - probability


        # Convert to percentage

        confidence_percentage = round(
            confidence * 100,
            2
        )


        # ----------------------------------------------------
        # Return result to web page
        # ----------------------------------------------------

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=confidence_percentage,
            filename=file.filename
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# Health check
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded"
    }


# ============================================================
# Run application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )