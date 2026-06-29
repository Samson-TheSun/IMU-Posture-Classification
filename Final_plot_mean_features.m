clc; clear; close all;

%% ============================================================
% MEAN FEATURE VISUALIZATION (STERNUM SENSOR ONLY)
% ============================================================
%
% DESCRIPTION:
% ------------------------------------------------------------
% This script performs visualization of axis-specific mean 
% features extracted from accelerometer data for posture analysis.
%
% INPUT:
% ------------------------------------------------------------
% The input to this script is the processed dataset:
%
%   → FINAL_DATASET_ACC.csv
%
% This dataset is generated from the main data processing pipeline, 
% which includes:
%
%   1. Raw accelerometer data acquisition
%   2. Signal preprocessing (smoothing)
%   3. Segmentation of stable posture regions (Z-axis variance)
%   4. Extraction of 45-second windows from each segment
%   5. Feature computation (Mean, Std, RMS, Magnitude, Tilt)
%
%
% DATA SELECTION (IMPORTANT):
% ------------------------------------------------------------
% The dataset contains data from two sensor locations:
%   - Sternum
%   - Belt
%
% In this script, ONLY sternum data is used:
%
%   → data.Sensor == 'sternum'
%
% This ensures:
%   - Consistent sensor orientation
%   - No sign inversion in gravity (Y-axis)
%   - Clean and interpretable feature distributions
%
%
% PURPOSE:
% ------------------------------------------------------------
% The goal is to visualize how posture classes are separated 
% based on axis-specific mean features:
%
%   - MeanX → Coronal plane (left/right movement)
%   - MeanY → Vertical axis (gravity, posture stability)
%   - MeanZ → Sagittal plane (forward/backward movement)
%
%
% OUTPUT:
% ------------------------------------------------------------
% The script generates and saves:
%
%   - MeanX_Across_Postures.jpg
%   - MeanY_Across_Postures.jpg
%   - MeanZ_Across_Postures.jpg
%
% These figures are used in Chapter 5 for feature interpretation.
%
% ============================================================

%% ============================================================
% Load Dataset
% ============================================================

data = readtable('FINAL_DATASET_ACC.csv');

%% ============================================================
% USE ONLY STERNUM DATA (IMPORTANT)
% ============================================================

data = data(strcmp(data.Sensor,'sternum'), :);

%% ============================================================
% Define Posture Order (IMPORTANT)
% ============================================================

order = {'A-1','A-2','A-3','A-4','A-5','A-6', ...
         'B-1','B-2','B-3','B-4','B-5','B-6'};

data.Posture = categorical(data.Posture, order, 'Ordinal', true);

%% ============================================================
% Plot MeanX
% ============================================================

figure;
boxplot(data.MeanX, data.Posture)
title('MeanX Across Postures')
xlabel('Posture')
ylabel('Mean X (m/s^2)')
grid on
set(gca,'FontSize',12)

saveas(gcf,'MeanX_Across_Postures.jpg')

%% ============================================================
% Plot MeanY
% ============================================================

figure;
boxplot(data.MeanY, data.Posture)
title('MeanY Across Postures')
xlabel('Posture')
ylabel('Mean Y (m/s^2)')
grid on
set(gca,'FontSize',12)

saveas(gcf,'MeanY_Across_Postures.jpg')

%% ============================================================
% Plot MeanZ
% ============================================================

figure;
boxplot(data.MeanZ, data.Posture)
title('MeanZ Across Postures')
xlabel('Posture')
ylabel('Mean Z (m/s^2)')
grid on
set(gca,'FontSize',12)

saveas(gcf,'MeanZ_Across_Postures.jpg')

disp('✅ Mean feature plots (sternum only) generated and saved!');