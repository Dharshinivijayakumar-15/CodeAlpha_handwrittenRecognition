import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==================================================
# STEP 1: LOAD MNIST DATASET
# ==================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training Images Shape:", x_train.shape)
print("Training Labels Shape:", y_train.shape)

# ==================================================
# STEP 2: NORMALIZE PIXEL VALUES
# Convert 0-255 --> 0-1
# ==================================================

x_train = x_train / 255.0
x_test = x_test / 255.0

# ==================================================
# STEP 3: ADD CHANNEL DIMENSION
# CNN expects:
# (height, width, channels)
# ==================================================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("New Shape:", x_train.shape)

# ==================================================
# STEP 4: BUILD CNN MODEL
# ==================================================

model = tf.keras.Sequential([

    # First Convolution Layer
    layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    # Reduce dimensions
    layers.MaxPooling2D((2,2)),

    # Second Convolution Layer
    layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ),

    # Reduce dimensions again
    layers.MaxPooling2D((2,2)),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation='relu'
    ),

    # Prevent overfitting
    layers.Dropout(0.3),

    # Output Layer (Digits 0-9)
    layers.Dense(
        10,
        activation='softmax'
    )
])

# ==================================================
# STEP 5: MODEL SUMMARY
# ==================================================

model.summary()

# ==================================================
# STEP 6: COMPILE MODEL
# ==================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================================================
# STEP 7: TRAIN MODEL
# ==================================================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1,
    batch_size=32
)

# ==================================================
# STEP 8: EVALUATE MODEL
# ==================================================

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

# ==================================================
# STEP 9: SAVE MODEL
# ==================================================

model.save("digit_cnn.keras")

print("\nModel saved successfully.")

# ==================================================
# STEP 10: PLOT ACCURACY
# ==================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Handwritten Digit Recognition")
plt.legend()

plt.savefig("accuracy_plot.png")
plt.show()