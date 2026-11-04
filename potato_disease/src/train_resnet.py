
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import load_datasets
from resnet18 import build_resnet18


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3

EPOCHS = 30
LEARNING_RATE = 0.001

MODEL_DIR = "models"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_final.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# Plot training history
# --------------------------------------------------

def plot_history(history):

    # Accuracy plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet-18 Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "resnet18_accuracy.png")
    )

    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-18 Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "resnet18_loss.png")
    )

    plt.close()


# --------------------------------------------------
# Main training function
# --------------------------------------------------

def train():

    # ----------------------------------------------
    # 1. Load datasets
    # ----------------------------------------------

    print("\nLoading datasets...")

    train_ds, val_ds, test_ds = load_datasets(
        image_size=IMAGE_SIZE
    )

    print("Datasets loaded successfully.")

    # ----------------------------------------------
    # 2. Build ResNet-18
    # ----------------------------------------------

    print("\nBuilding ResNet-18...")

    model = build_resnet18(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES
    )

    model.summary()

    # ----------------------------------------------
    # 3. Compile model
    # ----------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ----------------------------------------------
    # 4. Configure callbacks
    # ----------------------------------------------

    callbacks = [

        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        # Stop when validation loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when progress plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ----------------------------------------------
    # 5. Train model
    # ----------------------------------------------

    print("\nStarting ResNet-18 training...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=callbacks,
        verbose=1
    )

    # ----------------------------------------------
    # 6. Save final model
    # ----------------------------------------------

    model.save(FINAL_MODEL_PATH)

    print("\nFinal model saved:", FINAL_MODEL_PATH)
    print("Best model saved:", MODEL_PATH)

    # ----------------------------------------------
    # 7. Save training history
    # ----------------------------------------------

    plot_history(history)

    print("\nTraining plots saved.")

    print("\nResNet-18 training completed!")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    train()
