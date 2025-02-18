# Plant Disease Classification using CNN

## 📌 Project Overview

This project aims to classify plant diseases using Convolutional Neural Networks (CNNs). It utilizes TensorFlow and Keras for training a deep learning model on the PlantVillage dataset. The dataset consists of images of healthy and diseased plants, and the model predicts the class of an input image.

### 📂 Dataset

It contains multiple classes of plant diseases.

Images are loaded and preprocessed for model training.

## 🚀 Technologies Used

Python

TensorFlow / Keras

Matplotlib

NumPy

Pillow (PIL)

ImageDataGenerator for data augmentation


## ⚙️ Installation & Setup


**Install required dependencies:**

pip install tensorflow numpy pillow matplotlib

## 📊 Model Training

**The model is a Convolutional Neural Network (CNN) with the following layers:**

          Conv2D & MaxPooling layers for feature extraction.

          Flatten layer to convert features into a vector.

          Dense layers for classification.

          Softmax activation for multi-class output.

### Training is done using ImageDataGenerator:

    train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
      )

## 🔍 Image Prediction

To predict a plant disease from an image


## 🎯 Usage

Store plant images in the dataset folder.

Use main.py to classify new images.

Adjust img_size, batch_size, and epochs for better accuracy.

## 🛠 Future Improvements

Implement transfer learning for better accuracy.

Optimize model performance using hyperparameter tuning.

Deploy as a Streamlit Web App for real-time predictions.
