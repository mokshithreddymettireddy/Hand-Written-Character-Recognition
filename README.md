Handwritten Digit Recognition using CNN
This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to recognize handwritten digits from the MNIST dataset. The MNIST dataset contains 70,000 grayscale images of digits (0–9), each sized 28x28 pixels.
The trained model achieves high accuracy in classifying digits and also visualizes training progress through accuracy and loss plots.

📌 Features
Loads and preprocesses the MNIST dataset.
Builds a CNN with convolution, pooling, dropout, and dense layers.
Trains the model using categorical cross-entropy loss and Adam optimizer.
Evaluates accuracy on test data.
Plots training vs validation accuracy and loss.
Saves the trained model (mnist_cnn_model.h5) for future use.

🛠️ Requirements
Make sure you have the following installed (if not running in Colab):
pip install tensorflow numpy matplotlib

🚀 How to Run
⚡ Recommended: Run on Google Colab
Since this project involves training a deep learning model, it is best to run it on Google Colab to take advantage of free GPU acceleration.
Open Google Colab.
Upload the script (handwritten_digit_recognition.py) or copy the code into a new notebook.
Go to Runtime > Change runtime type > Hardware accelerator > GPU.
Run all cells.
Running Locally (Optional)
If you have a GPU-enabled machine, you can also run the script locally:
python handwritten_digit_recognition.py

📊 Model Architecture
Conv2D (32 filters, 3x3, ReLU)
MaxPooling2D (2x2)
Dropout (25%)
Conv2D (64 filters, 3x3, ReLU)
MaxPooling2D (2x2)
Dropout (25%)
Flatten
Dense (128 neurons, ReLU)
Dropout (50%)
Dense (10 neurons, Softmax)

📈 Results
Prints training and validation accuracy during training.
Final evaluation on test data shows:
Test Loss
Test Accuracy
Plots training history for accuracy and loss.

💾 Model Saving
The trained model is saved in HDF5 format:
mnist_cnn_model.h5
This can be loaded later for predictions without retraining.

🔍 Usage Example (Loading Saved Model)
To use the trained model later for predictions:

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
# Load the trained model
model = load_model('mnist_cnn_model.h5')
# Example: Predict on a single test image
from tensorflow.keras.datasets import mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255.0
# Pick one sample
sample = x_test[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample)
predicted_label = np.argmax(prediction)
plt.imshow(x_test[0].reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_label}")
plt.show()

🔮 Future Enhancements
Implement real-time digit recognition using a webcam.
Deploy the model as a Flask/Django web app.
Experiment with deeper architectures (e.g., ResNet, EfficientNet).
