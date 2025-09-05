import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

print("--- Handwritten Digit Recognition Project ---")
print("TensorFlow version:", tf.__version__)
# Load the dataset from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Define image dimensions and number of classes
img_rows, img_cols = 28, 28
num_classes = 10
# Reshape the data to fit the model's expected input shape (batch, height, width, channels)
# Add a channel dimension (1 for grayscale images)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# Normalize pixel values from the range [0, 255] to [0.0, 1.0]
# This improves training performance
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Convert class vectors to binary class matrices (one-hot encoding)
# e.g., for 10 classes, the integer '5' becomes a vector [0,0,0,0,0,1,0,0,0,0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print("\nData preprocessing complete.")
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"{x_train.shape[0]} train samples")
print(f"{x_test.shape[0]} test samples")
# Initialize a sequential model
model = Sequential()
# Add layers to the model
# Convolutional Layer 1: 32 filters of size 3x3, ReLU activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# Max Pooling Layer 1: Reduces spatial dimensions
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer 1: Prevents overfitting by randomly setting 25% of inputs to 0
model.add(Dropout(0.25))
# Convolutional Layer 2: 64 filters
model.add(Conv2D(64, (3, 3), activation='relu'))
# Max Pooling Layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer 2
model.add(Dropout(0.25))
# Flatten Layer: Converts the 2D feature maps to a 1D vector
model.add(Flatten())
# Dense Layer (Fully Connected): 128 neurons
model.add(Dense(128, activation='relu'))
# Dropout Layer 3
model.add(Dropout(0.5))
# Output Layer: 10 neurons (one for each class), softmax activation for probability distribution
model.add(Dense(num_classes, activation='softmax'))
# Print a summary of the model's architecture
print("\nModel Architecture:")
model.summary()
model.compile(loss='categorical_crossentropy',   
              optimizer='adam',                  
              metrics=['accuracy'])            
# Define training parameters
batch_size = 128
epochs = 12
print("\n--- Starting Model Training ---")
# Fit the model to the training data
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
print("--- Model Training Complete ---")
print("\n--- Evaluating Model Performance ---")
# Evaluate the model on the test dataset
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]*100:.2f}%')
print("\n--- Plotting Training History ---")
plt.figure(figsize=(14, 5))
# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.suptitle('CNN Training and Validation Metrics', fontsize=16)
plt.show()
# Save the entire model to a single HDF5 file.
model.save('mnist_cnn_model.h5')
print("\nModel saved successfully as 'mnist_cnn_model.h5'")
print("--- Project Complete ---")
