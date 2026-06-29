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
% 1. LOAD DATASET
% ============================================================

data = readtable('FINAL_DATASET_ACC.csv');


%% ============================================================
% 2. FILTER: USE ONLY STERNUM SENSOR
% ============================================================

data = data(strcmp(data.Sensor,'sternum'), :);


%% ============================================================
% 3. DEFINE ORDER OF POSTURES
% ============================================================

order = {'A-1','A-2','A-3','A-4','A-5','A-6', ...
         'B-1','B-2','B-3','B-4','B-5','B-6'};

data.Posture = categorical(data.Posture, order, 'Ordinal', true);


%% ============================================================
% 4. PLOT MEANX (CORONAL PLANE)
% ============================================================

figure;
boxplot(data.MeanX, data.Posture, ...
    'Colors','b', ...
    'Symbol','r+', ...
    'Whisker',1.5)

title('MeanX Across Postures')
xlabel('Posture')
ylabel('Mean X (m/s^2)')

set(gca,'FontSize',12)
set(gcf,'Color',[0.9 0.9 0.9])
set(gca,'Color',[0.9 0.9 0.9])
grid off

saveas(gcf,'MeanX_Across_Postures.jpg')


%% ============================================================
% 5. PLOT MEANY (VERTICAL AXIS)
% ============================================================

figure;
boxplot(data.MeanY, data.Posture, ...
    'Colors','b', ...
    'Symbol','r+', ...
    'Whisker',1.5)

title('MeanY Across Postures')
xlabel('Posture')
ylabel('Mean Y (m/s^2)')

set(gca,'FontSize',12)
set(gcf,'Color',[0.9 0.9 0.9])
set(gca,'Color',[0.9 0.9 0.9])
grid off

saveas(gcf,'MeanY_Across_Postures.jpg')


%% ============================================================
% 6. PLOT MEANZ (SAGITTAL PLANE)
% ============================================================

figure;
boxplot(data.MeanZ, data.Posture, ...
    'Colors','b', ...
    'Symbol','r+', ...
    'Whisker',1.5)

title('MeanZ Across Postures')
xlabel('Posture')
ylabel('Mean Z (m/s^2)')

set(gca,'FontSize',12)
set(gcf,'Color',[0.9 0.9 0.9])
set(gca,'Color',[0.9 0.9 0.9])
grid off

saveas(gcf,'MeanZ_Across_Postures.jpg')


%% ============================================================
% COMPLETION MESSAGE
% ============================================================

disp('✅ Sternum mean feature plots generated successfully!');
