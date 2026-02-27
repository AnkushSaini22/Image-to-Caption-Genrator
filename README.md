# Image to Caption Generator

## Overview
The Image to Caption Generator is a deep learning project that automatically generates descriptive textual captions for given images. It leverages a Convolutional Neural Network (CNN) to extract features from images and a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units to generate sequential text. 

## Dataset
This project uses the **Flickr8k Dataset**, which contains 8,091 images, each paired with 5 different descriptive captions. 
* The dataset is imported directly using `kagglehub` (`adityajn105/flickr8k`).
* It provides a rich vocabulary and diverse scenes to train the model effectively.

## Model Architecture
The architecture follows an Encoder-Decoder framework:
1. **Image Feature Extractor (Encoder):** * Utilizes the pre-trained **VGG16** model (with the final classification layer removed).
   * Extracts a 4096-dimensional feature vector for each image.
2. **Text Sequence Processor:**
   * Text data is cleaned (converted to lowercase, special characters removed) and tokenized.
   * Uses an **Embedding Layer** to represent words as dense vectors.
   * Processes the sequences using an **LSTM Layer** to maintain context over time.
3. **Decoder:**
   * Merges the extracted image features and text features.
   * Passes the combined data through Dense layers.
   * Uses a `softmax` activation function to predict the probability of the next word in the sequence.

## Prerequisites
To run this project, you need Python 3.x and the following libraries installed:
* TensorFlow / Keras
* NumPy
* Pandas
* NLTK (Natural Language Toolkit)
* Matplotlib
* Pillow (PIL)
* tqdm
* kagglehub

You can install the required packages using pip:
```bash
pip install tensorflow numpy pandas nltk matplotlib pillow tqdm kagglehub
