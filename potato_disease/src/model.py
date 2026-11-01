import tensorflow as tf
from tensorflow.keras import layers,models
def build_model(input_shape=(128,128,3), num_classes=3):
    model =models.Sequential([

        #Input
        layers.Input(shape=input_shape),

        #convolution block 1
        layers.Conv2D(
            32,
            kernel_size=(3,3),
            padding="same",
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2,2)),

        #Convolution Block 2
        layers.Conv2D(
            64,
            kernel_size=(3,3),
            padding="same",
            activation="relu"
        ),
        layers.MaxPool2D(pool_size=(2,2)),

        #Convolution Block 3
        layers.Conv2D(
            128,
            kernel_size=(3,3),
            padding="same",
            activation="relu"
        ),
        layers.MaxPool2D(pool_size=(2,2)),

        #Classification Head
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model

if "__name__" == "__main__":
    model=build_model()
    model.summary()

    #Test forward propagation

    dummy_input=tf.random.normal((1,128,128,3))
    output = model(dummy_input)
    print("\nOutput shape:", output.shape)
    print("Predicted probabilities:", output.numpy())
    print("Probability sum:", tf.reduce_sum(output).numpy())

