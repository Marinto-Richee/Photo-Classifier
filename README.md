
# Photo Classifier using TensorFlow

This script demonstrates how to build, train, and save a Convolutional Neural Network (CNN) for classifying photos into two categories: "not_accepted" and "accepted" using TensorFlow and Keras.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Marinto-Richee/Photo-Classifier.git
   ```

2. Install the necessary dependencies:

   ```bash
   pip install tensorflow
   ```

3. Download the dataset and organize it in the specified directory `images`.

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

1. **Input Layer**
2. **Pooling Layer**
3. **Convolutional Layer**
 
4. **Pooling Layer**
 

5. **Flattening Layer**

6. **Dense Layer**
7. **Output Layer**

The model takes an input image of shape (128, 128, 3), applies convolutional and pooling layers to extract features, flattens the output, and passes it through dense layers to make the final binary classification prediction using a sigmoid activation function.

![diagram](https://github.com/Marinto-Richee/Photo-Classifier/assets/65499285/9389011c-4120-4ea7-85d5-df2713ae7965)

## Data Augmentation

Data augmentation is a technique used to diversify and expand the training dataset by applying various transformations to the existing data. This helps in enhancing the model's ability to generalize and improve its performance. In the provided script, data augmentation is achieved using the `ImageDataGenerator` class from TensorFlow's Keras.

### Augmentation Techniques

1. **Rescaling (`rescale`):**
   - The `rescale` parameter is set to `1./255`, which normalizes the pixel values of the images to the range [0, 1]. This is a standard preprocessing step.

2. **Rotation (`rotation_range`):**
   - Randomly rotates the image by a specified degree.

3. **Width and Height Shift (`width_shift_range` and `height_shift_range`):**
   - Shifts the image horizontally or vertically within a specified range.

4. **Shear Transformation (`shear_range`):**
   - Shears the image in a specified direction within a given range.

5. **Zoom (`zoom_range`):**
   - Zooms into the image by a random factor within a specified range.

6. **Horizontal Flip (`horizontal_flip`):**
   - Flips the image horizontally.

7. **Vertical Flip (`vertical_flip`):**
   - Flips the image vertically.

### Implementation

The `ImageDataGenerator` is instantiated with the `rescale` parameter set to `1./255`. During the training process, this generator is used to load the dataset and apply the specified augmentation techniques to the images. Each time a batch of data is fetched using the `image_dataset_from_directory` function, the generator applies these augmentations, creating a more diverse set of training examples.

Data augmentation is a critical step in training deep learning models, especially for image-related tasks. It allows the model to learn from a wider range of variations and orientations of the input data, ultimately improving its ability to generalize to unseen data.

## Training the CNN Model

Training a Convolutional Neural Network (CNN) involves optimizing the model's weights using a training dataset to minimize a defined loss function. In this script, we use the `model.fit` method to train the CNN for the specified number of epochs.

### Model Compilation

Before training, the model is compiled using the `compile` method, where the following configurations are set:

- **Optimizer (`Adam`):**
  - Adam optimizer is used with a learning rate of 0.001. Adam is an adaptive optimization algorithm that is well-suited for training deep neural networks.

- **Loss Function (`BinaryCrossentropy`):**
  - Binary Crossentropy loss is used, which is suitable for binary classification problems.

- **Metrics (`accuracy`):**
  - The model's performance is evaluated based on accuracy during training.

### Training Process

The `model.fit` method is then called to train the model:

- **Input Data (`train_dataset`):**
  - The training dataset (`train_dataset`) is used to provide input data for training the model.

- **Validation Data (`val_dataset`):**
  - The validation dataset (`val_dataset`) is used to evaluate the model's performance after each epoch.

- **Number of Epochs:**
  - The model is trained for a specified number of epochs (`num_epochs`), which is set to 10 in this script.

During training, the CNN learns to extract features from the input images, optimize its weights based on the defined loss function, and improve its predictive accuracy. The training process iterates through the entire training dataset multiple times (epochs), adjusting the model's parameters to minimize the loss.

### Saving the Trained Model

After training, the trained CNN model is saved to a file using the `save` method, with the filename `Photo_classifier_tf_V1.h5`. This saved model can later be loaded and used for making predictions on new unseen data.

Training a model is a critical step in the machine learning workflow, and it's essential to monitor the training process, evaluate model performance, and save the trained model for future use.

## License

This project is licensed under the [MIT License](LICENSE).
