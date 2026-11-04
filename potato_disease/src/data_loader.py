import tensorflow as tf
AUTOTUNE= tf.data.AUTOTUNE
from sklearn.model_selection import train_test_split
from pathlib import Path

#configuration
# DATASET_DIR = "/mnt/d/DATA/POTATO DISEASE/datasets"
DATASET_DIR = '/mnt/d/DATA/POTATO DISEASE/datasets'
IMAGE_SIZE = (128,128)
BATCH_SIZE=32
SEED = 42

CLASS_NAMES= ["EARLYBLT", "HEALTHY", "LATEBLT"]

def load_datasets(image_size=IMAGE_SIZE):
    image_paths = []
    image_labels = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(DATASET_DIR) / class_name
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in supported_extensions:
                image_paths.append(str(image_path))
                image_labels.append(label)


    # -----------------------------------------
    # 2. Split: 70% training, 30% temporary
    # -----------------------------------------
    train_paths, temp_paths, train_labels, temp_labels =(
        train_test_split(
            image_paths,
            image_labels,
            test_size=0.30,
            stratify=image_labels,
            random_state=SEED
        )
    )

    # -----------------------------------------
    # 3. Split temporary: 15% validation,
    #    15% testing
    # -----------------------------------------

    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.50,
            stratify=temp_labels,
            random_state=SEED
        )
    )
    print(f"Training images:   {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Testing images:    {len(test_paths)}")

    # -----------------------------------------
    # 4. Image preprocessing
    # -----------------------------------------

    def preprocess_image(image_path, label):
        image= tf.io.read_file(image_path)
        image= tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )
        image.set_shape((None, None, 3))
        image= tf.image.resize_with_pad(
            image,
            image_size[0],
            image_size[1]
        )
        image=tf.cast(image, tf.float32) / 255.0
        return image, label
    # 5. Create TensorFlow datasets
    def create_datasets(paths, labels, training=False):
        ds=tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )
        if training:
            ds=ds.shuffle(
                buffer_size=len(paths),
                seed=SEED
            )
        ds= ds.map(
            preprocess_image,
            num_parallel_calls=AUTOTUNE
        )

        ds=ds.batch(BATCH_SIZE)
        ds=ds.prefetch(AUTOTUNE)
        return ds
    # 6. Build all three datasets
    train_ds=create_datasets(
        train_paths,
        train_labels,
        training=True
    )
    val_ds=create_datasets(
        val_paths,
        val_labels
    )
    test_ds=create_datasets(
        test_paths,
        test_labels
    )
    return train_ds, val_ds, test_ds





    # # Training dataset
    # train_ds=tf.keras.utils.image_dataset_from_directory(
    #     DATASET_DIR,
    #     labels="inferred",
    #     label_mode='int',
    #     class_names=CLASS_NAMES,
    #     image_size=image_size,
    #     pad_to_aspect_ratio=True,
    #     batch_size=BATCH_SIZE,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=SEED,
    #     shuffle=True
    # )

    # #Validation dataset
    # val_ds=tf.keras.utils.image_dataset_from_directory(
    #     DATASET_DIR,
    #     labels="inferred",
    #     label_mode="int",
    #     class_names=CLASS_NAMES,
    #     image_size=image_size,
    #     pad_to_aspect_ratio=True,
    #     batch_size=BATCH_SIZE,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=SEED,
    #     shuffle=False,
    # )
    # # Normalize pixel values and prefetch batches
    # def preprocess(images, labels):
        
    #     images = tf.cast(images, tf.float32) / 255.0
    #     return images, labels
    # train_ds = train_ds.map(
    #     preprocess,
    #     num_parallel_calls=AUTOTUNE
    # ).prefetch(AUTOTUNE)
    # val_ds=val_ds.map(
    #     preprocess,
    #     num_parallel_calls =AUTOTUNE
    # ).prefetch(AUTOTUNE)

    # return train_ds, val_ds

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