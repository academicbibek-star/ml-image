import os 
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data_loader import load_datasets
from src.model import build_model

#configuration
EPOCHS= 20
LEARNING_RATE = 0.001
MODEL_DIR="models"
MODEL_PATH = os.path.join(MODEL_DIR,"potato_cnn.keras")

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

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")
    plt.show()

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

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss.png")
    plt.show()  

def train():
    #1. Load Datasets
    train_ds, val_ds = load_datasets(image_size=(224,224))

    #2. Build CNN
    model=build_model(input_shape=(224,224,3))

    #3. compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    #Display Architecture
    model.summary()

    #Create Model Directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    #4. Callbacks 
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ] 

    #5. Train
    history=model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    #6. Plot Training history
    plot_history(history)
    print("\n training completed. ")
    print(f"best model saved at: {MODEL_PATH}")

if __name__=="__main__":
    print("Starting training script...")

    train()

    print("Training script finished.")

