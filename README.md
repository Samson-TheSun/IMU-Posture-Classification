# IMU-Based Posture Classification

This repository contains MATLAB code developed as part of a Master's thesis on posture classification using wearable IMU sensors.

## Overview

The project implements a complete pipeline for posture classification:

1. Data Processing and Segmentation
2. Feature Extraction from accelerometer signals
3. Machine Learning classification
4. Feature visualization

## Files

### 01_Data_Processing_Segmentation_Feature_Extraction.m
Implements:
- Signal preprocessing
- Adaptive segmentation (variance-based)
- 45-second window extraction
- Feature computation
- Dataset creation

### 02_Model_Training_and_Evaluation.m
Implements:
- Classification models (Random Forest, KNN, Logistic Regression)
- 5-fold cross-validation
- Performance evaluation

### 03_Feature_Visualization.m
Implements:
- Visualization of MeanX, MeanY, MeanZ
- Feature interpretation plots

## Notes

- Only accelerometer data is used
- Sternum sensor is used for classification due to higher stability
- Results are evaluated using cross-validation and hold-out splits

## Author
Samson Fekade Badishe
