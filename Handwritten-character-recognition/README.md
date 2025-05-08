# Handwritten Character Recognition

This project implements a neural network for recognizing handwritten digits using TensorFlow and Keras with the MNIST dataset. The project demonstrates building, training, and evaluating a basic neural network model.

## Project Structure
- `notebook/Handwritten_character_recognition.ipynb`: Jupyter notebook with code for loading data, building, training, and evaluating the model.
- `requirements.txt`: Lists all necessary Python libraries.

## Model Architecture
The neural network has the following architecture:
1. **Flatten Layer**: Converts each 28x28 image to a 1D array.
2. **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation.
3. **Dropout Layer**: Includes dropout to reduce overfitting.
4. **Output Layer**: A softmax layer for classifying digits from 0 to 9.

## Installation
To set up the environment, install the necessary libraries:
```bash
pip install -r requirements.txt
```
## Usage 
To train and test the model, open and run all cells in the Jupyter notebook located in the notebook/ directory.

## Results
The trained model achieves a high level of accuracy on the MNIST dataset.

