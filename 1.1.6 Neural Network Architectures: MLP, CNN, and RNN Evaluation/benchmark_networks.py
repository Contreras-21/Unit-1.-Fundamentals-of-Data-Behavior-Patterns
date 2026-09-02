import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. DATA PREPARATION (MNIST DATASET)
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (from 0-255 to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- ARCHITECTURE BUILDER FUNCTIONS ---

def build_mlp():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    return model, "MLP (Dense)"

def build_cnn():
    # CNN requires 2D image shape with channel: (28, 28, 1)
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model, "CNN (Convolutional)"

def build_rnn():
    # RNN (LSTM) treats image as sequence: 28 time steps, 28 features
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.LSTM(64, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model, "RNN (LSTM)"

# --- TRAINING & COMPARISON LOOP ---

models_to_test = [build_mlp(), build_cnn(), build_rnn()]
histories = {}

for model, name in models_to_test:
    print(f"\n--- Training: {name} ---")
    
    x_train_input = x_train
    x_test_input = x_test
    
    if "CNN" in name:
        x_train_input = np.expand_dims(x_train, -1)
        x_test_input = np.expand_dims(x_test, -1)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train_input, y_train, epochs=3, validation_split=0.1, batch_size=64)
    
    loss, acc = model.evaluate(x_test_input, y_test, verbose=0)
    print(f"Test Accuracy ({name}): {acc*100:.2f}%")
    histories[name] = history.history['val_accuracy']

# --- VISUALIZATION OF RESULTS ---
plt.figure(figsize=(10, 6))
for name, val_acc in histories.items():
    plt.plot(val_acc, label=name)

plt.title('Architecture Comparison (Validation Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()