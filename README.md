# Cat vs Dog Image Classification 🐱🐶
![Figure_2](https://github.com/SAIPRONE/cat_vs_dog_classification/assets/95390348/b7d9840c-8acc-428f-bb54-33201c79ac95)

A deep learning project using Convolutional Neural Networks (CNNs) to classify images into two categories: cats and dogs.

## 📌 Features:
- Uses TensorFlow's Keras API for deep learning.
- Data augmentation to prevent overfitting.
- Implements callbacks like `ReduceLROnPlateau` for dynamic learning rate adjustments.
- Visualizes training and validation loss and accuracy.
  
## 🛠️ Dependencies:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [tensorflow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## 📂 Data:
The dataset consists of images of cats and dogs:
[Dogs vs. Cats](C:/Users/saibr/Desktop/ML/supervised-learning/jamk-cnn/input/train/train/)

## 🚀 How to Run:
1. Ensure you have all the required dependencies installed.
2. Download the dataset to your local directory.
3. Execute the Python script. It will:
   - Load the data
   - Display a random sample image
   - Create the CNN model
   - Split the data for training and validation
   - Train the model
   - Display training & validation loss and accuracy graphs.

## 📉 Model Architecture:
1. Three Convolutional layers each followed by Batch Normalization, MaxPooling, and Dropout.
2. A Flatten layer to transform the 3D feature maps to 1D feature vectors.
3. Two Dense layers - one for feature extraction and the other as the output layer.

## 🤔 Note:
The `FAST_RUN` variable is used to determine the number of epochs. If set to `True`, the script will run for 3 epochs. Otherwise, it'll run for 10 epochs.

## Author
**Fadi Helal**
## 📜 License:
This project is open-source and available under the MIT License.

## 🔗 Useful Resources:
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Introduction to CNNs](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks)
- [ImageDataGenerator for Data Augmentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
