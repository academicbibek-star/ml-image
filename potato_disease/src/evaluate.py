import os 
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from data_loader import load_datasets

MODEL_PATH = Path("models/potato_cnn.keras")
TEST_DIR =Path("potato_test")
RESULTS_DIR=Path("results")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES= [
    "EARLYBLT",
    "HEALTHY",
    "LATEBLT"
]

TEST_DIR.mkdir(parents=True, exist_ok=True)

def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CLASS_NAMES,
        shuffle=False
    )

    #Important
    #Normalize Only if the training pipeline is normalized otherwise the result 
    # will be inaccurate or random
    test_ds = test_ds.map(
        lambda images, labels : (images / 255.0 , labels)
    )

    return test_ds

#-------------------------------
# Evaluate model
#-------------------------------

def evaluate_model():
    print("Loading trained model....")
    model=tf.keras.models.load_model(MODEL_PATH)
    print("Loading test dataset...")
    # test_ds = load_test_dataset()
    _, _, test_ds = load_datasets()
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )
    print("\nTest Results")
    print("-------------------------")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])
    predictions = model.predict(test_ds, verbose = 1)
    y_pred = np.argmax(predictions, axis=1)

    #-------------------------------------------------
    # Classification Report
    #-------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        laels = np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print("\n CLassification Report")
    print("--------------------------------")
    print(report)

    with open(
        RESULTS_DIR /"classification_report.txt",
        "w"
    )as file:
        file.write(report)

    cm=confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES))
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Precited Label")
    plt.ylabel("Actual Label")
    plt.title("Potato Disease Classification")

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR/"confusion_matrix.png"
    )
    plt.close()
    print("\nConfusion matrix saved.")

    print("\nEvaluation completed!")

if "__name__"=="__main__":
    evaluate_model()



