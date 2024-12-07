# math5836-project2-Abalone-Age-Prediction

This repository contains the solution to predicting the age of abalone (marine mollusks) from physical measurements, using both regression and classification models. The age of an abalone is traditionally determined by counting the rings in its shell, a time-consuming task. This project uses alternative measurements to predict the age of abalones efficiently.

# Project Overview
The goal of this project is to predict the ring-age of abalones (which is the number of rings on the shell) based on various physical features such as length, diameter, weight, and others. The project also includes a classification task to categorize abalones into two age groups: below 7 years and above 7 years.

# Dataset
The dataset used for this project is from the UCI Machine Learning Repository. You can find more details here. The dataset contains the following attributes:

Sex: Nominal (M = Male, F = Female, I = Infant)
Length: Continuous (mm) – Longest shell measurement
Diameter: Continuous (mm) – Perpendicular to the length
Height: Continuous (mm) – With meat in shell
Whole weight: Continuous (grams) – Whole abalone
Shucked weight: Continuous (grams) – Weight of meat
Viscera weight: Continuous (grams) – Gut weight (after bleeding)
Shell weight: Continuous (grams) – After drying
Rings: Integer – Number of rings (used to determine age in years, but this project uses the raw value without the "+1.5" adjustment)

# Data Processing
The data processing tasks include:

Data Cleaning: Convert categorical values such as 'M' and 'F' to numerical values (0 and 1).
Correlation Analysis: Generate a heatmap to visualize the correlations between features, identify the two most correlated features with the ring-age, and create scatter plots.
Data Splitting: Split the data into a training set (60%) and a test set (40%) using a random seed.
Visualizations: Create various visualizations like histograms and scatter plots to explore relationships between features.

# Models Developed
## 1. Linear Regression (Ring Age Prediction)
A linear regression model is trained on the dataset to predict the ring-age (continuous variable) using all available features. Performance metrics include:

RMSE (Root Mean Squared Error)
R-squared score
## 2. Logistic Regression (Age Classification)
A logistic regression model classifies abalones into two categories based on their ring-age: below 7 years or above 7 years. Performance metrics for the classification model include:
Accuracy
AUC (Area Under the Curve)
ROC Curve

## 3. Model Comparisons
Comparison with and without feature normalization: We compare linear and logistic regression models with normalized and non-normalized features.
Feature Selection: Models are also tested using only two selected features that are most strongly correlated with ring-age.

## 4. Neural Network (SGD Optimized)
A neural network model is trained using Stochastic Gradient Descent (SGD). Hyperparameters such as the number of layers, number of neurons per layer, and learning rate are tuned to optimize performance. The neural network model is then compared to the best-performing linear model.

## 5. Experimental Results
The models are evaluated over 30 experiments to get the mean and standard deviation for RMSE and R-squared scores on both the training and test datasets.

