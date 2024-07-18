KNN Image Classification with OpenCV and CV Studio

Overview
This project demonstrates how to train a K-Nearest Neighbours (k-NN) classifier to classify cat and dog images using OpenCV and Computer Vision Learning Studio (CV Studio). The project includes the following steps:

Importing necessary libraries.
Loading and processing images.
Training a k-NN classifier.
Evaluating the classifier using accuracy and confusion matrix.
Saving and reporting the model back to CV Studio.
Objectives
Classify images of cats and dogs using the k-NN algorithm.
Use OpenCV for image processing and k-NN implementation.
Utilize CV Studio for dataset management and model reporting.
Requirements
Python 3.x
OpenCV
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
imutils
skillsnetwork
Steps
1. Import Libraries
Import necessary libraries for data processing, visualization, image processing, and machine learning.

2. Download Images and Annotations
Initialize the CV Studio client and download the images and annotations for the project.

3. Load and Plot Images
Read and plot random images from the dataset to visualize the data. Convert images to grayscale and resize them for processing.

4. Process All Images
Load and process all images, converting them to grayscale, resizing, and flattening them. Label each image based on the annotations.

5. Train the k-NN Model
Split the data into training and test sets. Train the k-NN model using different values of k and evaluate the model's performance.

6. Evaluate the Model
Calculate the accuracy and create a confusion matrix for each value of k to determine the best k.

7. Save and Report the Model
Save the trained model and report the results back to CV Studio.

Results
The best value of k for the k-NN classifier is determined based on the highest accuracy.
Confusion matrices are created to evaluate the model's performance.
Conclusion
The project successfully demonstrates how to use the k-NN algorithm for image classification using OpenCV and CV Studio. The best k value and model performance are reported back to CV Studio for further analysis
