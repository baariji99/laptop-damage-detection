import sys
import keras
import numpy as np

MODEL_PATH = "laptop_screen_damage_model.keras"
IMG_SIZE = (224, 224)

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

def predict_image(image_path):
    image = keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

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

    print("\nPrediction")
    print("-" * 40)
    print("Image:", image_path)
    print("Result:", prediction)
    print("Confidence:", f"{confidence * 100:.2f}%")
    print("Raw probability:", f"{probability:.4f}")


if len(sys.argv) != 2:
    print("\nUsage:")
    print("python predict.py <image_path>")
    sys.exit(1)

predict_image(sys.argv[1])