import tensorflow as tf
AUTOTUNE= tf.data.AUTOTUNE

#configuration
DATASET_DIR = "dataset"
IMAGE_SIZE = (128,128)
BATCH_SIZE=32
SEED = 42

CLASS_NAMES= ["early_blt", "healthy", "late_blt"]

def load_datasets():
    # Training dataset
    train_ds=tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        shuffle=True
    )

    #Validation dataset
    val_ds=tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        shuffle=False,
    )
    # Normalize pixel values and prefetch batches
    def preprocess(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels
    train_ds = train_ds.map(
        preprocess,
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    val_ds=val_ds.map(
        preprocess,
        num_parallel_calls =AUTOTUNE
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds

if __name__== '__main__':
    train_ds,val_ds=load_datasets()
    print(f"Classes: {CLASS_NAMES}")
    for images, labels in train_ds.take(1):
        print("Image batch shape: ", images.shape)
        print("Label batch shape: ", labels.shape)
        print(f"""
              Pixel range: 
              {tf.reduce_min(images).numpy()},
              {tf.reduce_max(images).numpy()} 
              """)