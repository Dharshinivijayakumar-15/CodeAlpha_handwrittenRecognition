# ============ IMPORTS ============
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# ============ LOAD SAVED MODEL ============
model = keras.models.load_model('digit_cnn.keras')
print("Model loaded successfully!")

# ============ LOAD TEST DATA ============
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# One hot encode
y_test = keras.utils.to_categorical(y_test, 10)

# ============ TEST WITH RANDOM IMAGES ============
indices = np.random.randint(0, len(x_test), 10)
test_images = x_test[indices]
test_labels = y_test[indices]

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(test_labels, axis=1)

# ============ SHOW PREDICTIONS ============
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i].reshape(28, 28), cmap='gray')
    color = 'green' if predicted_labels[i] == actual_labels[i] else 'red'
    ax.set_title(f'Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}',
                 color=color)
    ax.axis('off')
plt.suptitle('CNN Model Predictions — Green=Correct, Red=Wrong')
plt.tight_layout()
plt.savefig('predictions.png')
plt.show()

print("\n=== PREDICTION RESULTS ===")
for i in range(10):
    status = "✅ Correct" if predicted_labels[i] == actual_labels[i] else "❌ Wrong"
    print(f"Image {i+1}: Actual={actual_labels[i]}, Predicted={predicted_labels[i]} {status}")