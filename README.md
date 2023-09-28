
# Photo Classifier using TensorFlow

This script demonstrates how to build, train, and save a Convolutional Neural Network (CNN) for classifying photos into two categories: "not_accepted" and "accepted" using TensorFlow and Keras.

## Getting Started

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Install the necessary dependencies:

   ```bash
   pip install tensorflow
   ```

3. Download the dataset and organize it in the specified directory `ID Card Program/organized_photos - Copy`.

4. Modify the `data_dir` variable in the script to point to the dataset directory.

## Usage

1. Run the script to train the CNN model and save it to a file:

   ```bash
   python script.py
   ```

2. The trained model will be saved as `Photo_classifier_tf_V1.h5`.

## Directory Structure

```
|-- ID Card Program
|   |-- organized_photos - Copy
|       |-- not_accepted
|           |-- image1.jpg
|           |-- image2.jpg
|           ...
|       |-- accepted
|           |-- image1.jpg
|           |-- image2.jpg
|           ...
|-- script.py
|-- Photo_classifier_tf_V1.h5
|-- README.md

```
## Model Architecture

The Convolutional Neural Network (CNN) model is designed for image classification. It comprises the following layers:

1. **Input Layer:**
   - Type: Conv2D
   - Filters: 32
   - Kernel Size: (3, 3)
   - Activation Function: ReLU
   - Padding: Same
   - Input Shape: (128, 128, 3)

2. **Pooling Layer:**
   - Type: MaxPooling2D
   - Pool Size: (2, 2)

3. **Convolutional Layer:**
   - Type: Conv2D
   - Filters: 64
   - Kernel Size: (3, 3)
   - Activation Function: ReLU
   - Padding: Same

4. **Pooling Layer:**
   - Type: MaxPooling2D
   - Pool Size: (2, 2)

5. **Flattening Layer:**
   - Flattens the input to feed into the dense layers.

6. **Dense Layer:**
   - Neurons: 128
   - Activation Function: ReLU

7. **Output Layer:**
   - Neurons: 1 (for binary classification)
   - Activation Function: Sigmoid

The model takes an input image of shape (128, 128, 3), applies convolutional and pooling layers to extract features, flattens the output, and passes it through dense layers to make the final binary classification prediction using a sigmoid activation function.

![diagram](https://github.com/Marinto-Richee/Photo-Classifier/assets/65499285/9389011c-4120-4ea7-85d5-df2713ae7965)


## Data Augmentation

The dataset is augmented using the rescaling of pixel values to improve training performance.

## Training

The model is compiled using the Adam optimizer with a learning rate of 0.001 and Binary Crossentropy loss. It is then trained for 10 epochs using the training dataset and validated using the validation dataset.

## License

This project is licensed under the [MIT License](LICENSE).
