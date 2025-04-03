# ELE489_MachineLearning_HW1
KNN Classification Project:
This project implements the k-Nearest Neighbors (KNN) algorithm to analyze how different parameters affect classification accuracy. Some of the key features of the project are
it compares Euclidean and Manhattan distance metrics, evaluates the impact of k-values (from 1 to 25+). Also, includes data normalization (min-max scaling) to ensure fair distance comparisons

My comments of outputs:
Small k (e.g., 1-3): Higher sensitivity to noise/outliers (risk of overfitting)
Moderate k (e.g., 5-9): Optimal balance (best accuracy in this project)
Large k (e.g., 25+): May include irrelevant neighbors, reducing accuracy (underfitting)

Output
Accuracy comparisons for different k values and distance metrics.
Visualization of results.

How to Run

Prerequisites:
Libraries: numpy, pandas, scikit-learn, matplotlib

NOTE:To access the data set in the code, you need to download the data 
file from the link below and copy it to the required file path. 
I also added a section containing the data link as a comment to the code so that you can run the code without the file.
You can also run the code using this section.

LINK:https://archive.ics.uci.edu/dataset/109/wine
