
import tensorflow as tf
from tensorflow.keras import layers, Model


def residual_block(
    x,
    filters,
    stride=1,
    name=None
):
    """
    Basic ResNet residual block.

    Args:
        x: Input tensor
        filters: Number of output channels
        stride: Convolution stride
        name: Optional block name
    """

    shortcut = x

    # First convolution
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu1"
    )(x)

    # Second convolution
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    # Projection shortcut when dimensions differ
    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name}_shortcut_conv"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_shortcut_bn"
        )(shortcut)

    # Residual addition
    x = layers.Add(
        name=f"{name}_add"
    )([x, shortcut])

    # Final activation
    x = layers.ReLU(
        name=f"{name}_relu2"
    )(x)

    return x


def build_resnet18(
    input_shape=(224, 224, 3),
    num_classes=3
):

    inputs = layers.Input(shape=input_shape)

    # ----------------------------------
    # Initial convolution (Stem)
    # ----------------------------------

    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1"
    )(inputs)

    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
        name="maxpool"
    )(x)

    # ----------------------------------
    # Stage 1: 64 filters
    # ----------------------------------

    x = residual_block(
        x, filters=64, stride=1, name="stage1_block1"
    )

    x = residual_block(
        x, filters=64, stride=1, name="stage1_block2"
    )

    # ----------------------------------
    # Stage 2: 128 filters
    # ----------------------------------

    x = residual_block(
        x, filters=128, stride=2, name="stage2_block1"
    )

    x = residual_block(
        x, filters=128, stride=1, name="stage2_block2"
    )

    # ----------------------------------
    # Stage 3: 256 filters
    # ----------------------------------

    x = residual_block(
        x, filters=256, stride=2, name="stage3_block1"
    )

    x = residual_block(
        x, filters=256, stride=1, name="stage3_block2"
    )

    # ----------------------------------
    # Stage 4: 512 filters
    # ----------------------------------

    x = residual_block(
        x, filters=512, stride=2, name="stage4_block1"
    )

    x = residual_block(
        x, filters=512, stride=1, name="stage4_block2"
    )

    # ----------------------------------
    # Classification head
    # ----------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_avg_pool"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet18"
    )

    return model

if __name__ == "__main__":

    model = build_resnet18()

    model.summary()

    print("\nTotal parameters:", model.count_params())