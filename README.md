# CNN-Based Image Classification on MNIST Dataset

## 📜 Project Overview
This project involves building and evaluating a Convolutional Neural Network (CNN) to classify images from the MNIST dataset, which consists of handwritten digits (0-9). The objective is to understand CNN architecture, achieve high accuracy, and explore the practical applications of CNN in image processing and workflow automation.

---

## 🚀 Business Use Cases
- **Image Processing**: Automate classification of handwritten digits.
- **Workflow Automation**: Enable efficient digit recognition for applications like automated form processing and postal code recognition.

---

## ⚙️ Project Workflow

### 1. **Dataset Loading and Exploration**
- The MNIST dataset is loaded using TensorFlow's `tf.keras.datasets`.
- Visualized a sample of the dataset using Matplotlib.
- Performed a brief analysis of class distributions to ensure balanced data.

### 2. **Data Preprocessing**
- Normalized image pixel values to a range of 0 to 1.
- Converted class labels into one-hot encoded format for multi-class classification.
- Split the dataset into training, validation, and test sets.

### 3. **Data Augmentation**
- Applied data augmentation techniques such as rotation, flipping, and zooming using TensorFlow's `ImageDataGenerator` to increase dataset diversity and improve model generalization.

### 4. **Build the CNN Model**
- Designed a CNN architecture with the following layers:
  - **Input Layer**: Takes in the normalized images.
  - **Convolutional Layers**: Extract features using filters and ReLU activation.
  - **MaxPooling Layers**: Downsample feature maps.
  - **Flatten Layer**: Converts 2D feature maps into 1D.
  - **Dense Layers**: Perform classification with softmax activation at the output.
  - **Dropout Layers**: Reduce overfitting.
- Selected the **Adam** optimizer and **ReLU** activation function for efficient learning.

### 5. **Compile and Train the Model**
- Compiled the model using `categorical_crossentropy` as the loss function for multi-class classification.
- Trained the model on the training set while validating on the validation set.
- Used callbacks like early stopping and learning rate reduction for efficient training.

### 6. **Model Evaluation**
- Evaluated the trained model on the test set.
- Calculated the following metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
- Plotted confusion matrices and learning curves for better insights into the model's performance.

### 7. **Save and Deploy the Model**
- Saved the trained model for future use using TensorFlow's `save` functionality.
- Deployment-ready model for potential cloud hosting (e.g., AWS, GCP).

---

## 📊 Results
- Achieved high accuracy, precision, recall, and F1 score with minimal loss.
- Generated a detailed confusion matrix to understand class-wise performance.

---

## 🧮 Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Binary Cross-Entropy Loss**

---

## 📂 Dataset
The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale, with dimensions 28x28 pixels.

Dataset Source: [TensorFlow Datasets - MNIST](https://www.tensorflow.org/datasets)

---

## 🛠️ Technical Tags
- **Image Processing**
- **CNN**
- **Classification**
- **TensorFlow/Keras**
- **Deep Learning**

---
