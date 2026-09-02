import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

print("Loading the dataset and training the models...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

def get_mlp():
    model = models.Sequential([layers.Flatten(input_shape=(28, 28)), layers.Dense(128, activation='relu'), layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_cnn():
    model = models.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(64, activation='relu'), layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_rnn():
    model = models.Sequential([layers.Input(shape=(28, 28)), layers.LSTM(64, activation='tanh'), layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

mlp, cnn, rnn = get_mlp(), get_cnn(), get_rnn()
mlp.fit(x_train, y_train, epochs=2, verbose=0)
cnn.fit(np.expand_dims(x_train, -1), y_train, epochs=2, verbose=0)
rnn.fit(x_train, y_train, epochs=2, verbose=0)

filename = 'numero.png'
if not os.path.exists(filename):
    print(f"ERROR: The file '{filename}' was not found.")
else:
    image = Image.open(filename).convert('L')
    image = ImageOps.invert(image).resize((28, 28))
    image_array = np.array(image) / 255.0
    plt.imshow(image_array, cmap='gray')
    plt.title('Processed input (28x28)')
    plt.axis('off')
    plt.show()
    mlp_prediction = np.argmax(mlp.predict(np.expand_dims(image_array, 0)), axis=-1)[0]
    cnn_input = image_array.reshape(1, 28, 28, 1)
    cnn_prediction = np.argmax(cnn.predict(cnn_input), axis=-1)[0]
    rnn_prediction = np.argmax(rnn.predict(np.expand_dims(image_array, 0)), axis=-1)[0]
    print(f"MLP prediction: {mlp_prediction}")
    print(f"CNN prediction: {cnn_prediction}")
    print(f"RNN prediction: {rnn_prediction}")