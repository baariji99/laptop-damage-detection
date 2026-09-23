import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
TEST_DIR = "test"

MODEL_FILE = "laptop_screen_damage_model.keras"

print("=" * 60)
print("LAPTOP SCREEN DAMAGE DETECTION")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

validation_ds = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# ============================================================
# CHECK CLASSES
# ============================================================

class_names = train_ds.class_names

print("\nClasses:", class_names)

if set(class_names) != {"broken", "normal"}:
    raise ValueError(
        f"Expected classes ['broken', 'normal'], but found {class_names}"
    )

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ============================================================
# MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(224, 224, 3),
    name="image"
)

x = data_augmentation(inputs)

# MobileNetV2 expects inputs in [-1, 1]
x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="damage_probability"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="LaptopScreenDamageDetector"
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        ),
        keras.metrics.AUC(
            name="auc"
        )
    ]
)

print("\nModel created successfully.")

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = keras.models.load_model(
    MODEL_FILE
)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_results = model.evaluate(
    validation_ds,
    verbose=1
)

for metric_name, value in zip(
    model.metrics_names,
    validation_results
):
    print(f"{metric_name}: {value:.4f}")

# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_results = model.evaluate(
    test_ds,
    verbose=1
)

for metric_name, value in zip(
    model.metrics_names,
    test_results
):
    print(f"{metric_name}: {value:.4f}")

# ============================================================
# FINAL SAVE
# ============================================================

model.save(MODEL_FILE)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Model saved as: {MODEL_FILE}")
print("Classes:", class_names)
print("Input size:", IMG_SIZE)