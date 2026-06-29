clc; clear; close all;
%% ============================================================
% FINAL MASTER CODE
% ACCELEROMETER-BASED SEGMENTATION, FEATURE EXTRACTION, AND DATASET CREATION
% ============================================================

% DESCRIPTION:
% ------------------------------------------------------------
% This program implements a complete data processing pipeline 
% for posture classification using wearable sensor data.
%
% The pipeline begins with raw accelerometer signals collected 
% from two sensor positions: sternum and belt. The primary objective 
% is to identify stable posture segments, extract meaningful features, 
% and construct a structured dataset suitable for machine learning.
%
% METHODOLOGY:
% ------------------------------------------------------------
% 1. Data Loading:
%    Raw accelerometer data (acc_stream.csv) is loaded for each 
%    participant and sensor position.
%
% 2. Signal Preprocessing:
%    The raw signals are smoothed using a moving average filter 
%    to reduce noise and improve stability detection.
%
% 3. Segmentation:
%    Stable posture segments are identified using a variance-based 
%    approach applied to the Z-axis of the accelerometer signal.
%    Regions with low variance are assumed to correspond to static 
%    postures.
%
% 4. Segment Selection:
%    Valid segments are filtered based on duration, and the 12 
%    most representative segments are selected for each participant. 
%    These segments are mapped to predefined posture labels 
%    (A-1 to B-6).
%
% 5. Window Extraction:
%    From each selected segment, a 45-second window (centered 
%    within the segment) is extracted to ensure consistency in 
%    feature computation.
%
% 6. Visualization:
%    Segmentation results are visualized to validate correctness.
%    Full segments and corresponding 45-second windows are plotted 
%    for comparison.
%
% 7. Feature Extraction:
%    A set of statistical and orientation-based features are computed:
%       - Mean (X, Y, Z): Represents average signal level
%       - Standard deviation: Captures variability
%       - RMS: Represents signal energy
%       - Magnitude features: Combined 3-axis information
%       - Tilt features: Represent body orientation relative to gravity
%
% 8. Dataset Construction:
%    All computed features are stored along with metadata:
%       - Participant ID
%       - Location (Oslo)
%       - Sensor position (sternum/belt)
%       - Posture label (A-1 to B-6)
%
% 9. Data Export:
%    The final dataset is saved in multiple formats:
%       - CSV (for machine learning)
%       - Excel (.xlsx)
%       - MATLAB (.mat)
%
% DESIGN CHOICE:
% ------------------------------------------------------------
% Only accelerometer data is used in this study. This avoids 
% mixing different sensor modalities (e.g., gyroscope and 
% magnetometer), which can introduce scaling inconsistencies 
% and degrade classification performance.
%
% OUTPUT:
% ------------------------------------------------------------
% - FINAL_DATASET_ACC.csv
% - FINAL_DATASET_ACC.xlsx
% - FINAL_DATASET_ACC.mat
% - Segmentation plots (verification)
% - Full vs 45-second comparison plots
%
% This dataset serves as input for subsequent machine learning analysis.
% ============================================================
%% ============================================================
% SECTION 1 — SETTINGS
% ============================================================

base_path = 'C:\Users\badissx\Documents\UIO\Mast-Pro\MT_Samson_John-Fred\Posture Orginal - CSV\MT_DATA\Final_Master';

true_order = ["A-1","A-2","A-3","A-4","A-5","A-6", ...
              "B-1","B-2","B-3","B-4","B-5","B-6"];

positions = ["sternum","belt"];

Fs = 50;
Nwin = 45 * Fs;

FINAL_DATA = table();

%% ============================================================
% LOOP PARTICIPANTS
% ============================================================

for p = 1:20
    
    folder = sprintf('P%02d', p);
    fprintf('\nProcessing %s...\n', folder);

    %% ========================================================
    % 1. LOAD STERNUM ACC DATA
    %% ========================================================

    file_st = fullfile(base_path, folder, 'sternum','acc_stream.csv');
    if ~isfile(file_st), continue; end
    
    data = readtable(file_st);
    t = data.timestamp;
    
    z = movmean(data.z,80);
    N = length(z);

    %% ========================================================
    % 2. DETECT STABLE SEGMENTS
    %% ========================================================

    stable = zeros(N,1);
    
    for i = 1:N-200
        if var(z(i:i+200)) < 0.02
            stable(i:i+200) = 1;
        end
    end

    d = diff([0; stable; 0]);
    start_idx = find(d==1);
    end_idx   = find(d==-1)-1;
    segments = [start_idx end_idx];

    %% ========================================================
    % 3. FILTER + SELECT 12 SEGMENTS
    %% ========================================================

    valid_segments = [];
    for i = 1:size(segments,1)
        if (segments(i,2)-segments(i,1)) > 500
            valid_segments = [valid_segments; segments(i,:)];
        end
    end

    if size(valid_segments,1) < 12
        L = floor(N/12);
        selected = [];
        for k = 1:12
            selected = [selected; (k-1)*L+1, min(k*L,N)];
        end
    else
        Lg = valid_segments(:,2)-valid_segments(:,1);
        [~,idx] = sort(Lg,'descend');
        selected = valid_segments(idx(1:12),:);
        selected = sortrows(selected,1);
    end

    %% ========================================================
    % 4. SEGMENTATION PLOT (CONFIRMATION)
    %% ========================================================

    fig = figure('Visible','off','Position',[100 100 1400 400]);
    plot(t,z,'b'); hold on

    for i = 1:12
        idx1 = selected(i,1);
        idx2 = selected(i,2);

        patch([t(idx1) t(idx2) t(idx2) t(idx1)], ...
              [min(z) min(z) max(z) max(z)], ...
              'green','FaceAlpha',0.15,'EdgeColor','none')

        xline(t(idx1),'g','LineWidth',2)
        xline(t(idx2),'r--','LineWidth',1.5)

        mid = floor((idx1+idx2)/2);
        text(t(mid), max(z), true_order(i), ...
            'HorizontalAlignment','center','FontWeight','bold')
    end

    title(sprintf('Adaptive Segmentation - %s (Sternum ACC)',folder))
    xlabel('Time'); ylabel('Z-axis')

    saveas(fig, fullfile(base_path, folder,...
        sprintf('Segmentation_%s.png',folder)))
    close(fig)

    %% ========================================================
    % 5. FULL vs 45s PLOTS
    %% ========================================================

    for pos = positions
        
        file_path = fullfile(base_path, folder, pos,'acc_stream.csv');
        if ~isfile(file_path), continue; end
        
        data = readtable(file_path);
        t = data.timestamp;
        z = movmean(data.z,80);

        fig = figure('Visible','off','Position',[100 100 1200 2000]);

        for i = 1:12
            
            idx1 = selected(i,1);
            idx2 = selected(i,2);
            mid = floor((idx1+idx2)/2);

            i1 = max(1, mid-Nwin/2);
            i2 = min(length(z), mid+Nwin/2);

            posture = true_order(i);

            % FULL
            subplot(12,2,(i-1)*2+1)
            plot(t(idx1:idx2), z(idx1:idx2),'b')
            title(sprintf('%s - %s (Full)',folder,posture))

            % 45s WINDOW
            subplot(12,2,(i-1)*2+2)
            plot(t(i1:i2), z(i1:i2),'r')
            title(sprintf('%s - %s (45s)',folder,posture))
        end

        sgtitle(sprintf('Full vs 45s - %s (%s)',folder,pos))

        saveas(fig, fullfile(base_path, folder,...
            sprintf('FullVs45_%s_%s.png',folder,pos)))

        close(fig)
    end

    %% ========================================================
    % 6. FEATURE EXTRACTION (ACC ONLY)
    %% ========================================================

    for pos = positions

        file_path = fullfile(base_path, folder, pos,'acc_stream.csv');
        if ~isfile(file_path), continue; end

        data = readtable(file_path);
        x = movmean(data.x,80);
        y = movmean(data.y,80);
        z = movmean(data.z,80);

        for i = 1:12

            idx1 = selected(i,1);
            idx2 = selected(i,2);
            mid = floor((idx1+idx2)/2);

            i1 = max(1, mid-Nwin/2);
            i2 = min(length(x), mid+Nwin/2);

            if (i2-i1) < 1000, continue; end

            X = x(i1:i2);
            Y = y(i1:i2);
            Z = z(i1:i2);

            % FEATURES
            MeanX=mean(X); MeanY=mean(Y); MeanZ=mean(Z);
            StdX=std(X); StdY=std(Y); StdZ=std(Z);
            RMSX=rms(X); RMSY=rms(Y); RMSZ=rms(Z);

            Mag = sqrt(X.^2+Y.^2+Z.^2);
            MagMean=mean(Mag); MagStd=std(Mag); MagRMS=rms(Mag);

            Tilt = atan2(sqrt(X.^2+Y.^2),Z);
            TiltMean=mean(Tilt); TiltStd=std(Tilt);

            row = table({folder},{'Oslo'},{char(pos)},...
                {char(true_order(i))},...
                MeanX,MeanY,MeanZ,...
                StdX,StdY,StdZ,...
                RMSX,RMSY,RMSZ,...
                MagMean,MagStd,MagRMS,...
                TiltMean,TiltStd,...
                'VariableNames',...
                {'Participant','Location','Sensor','Posture',...
                 'MeanX','MeanY','MeanZ',...
                 'StdX','StdY','StdZ',...
                 'RMSX','RMSY','RMSZ',...
                 'MagMean','MagStd','MagRMS',...
                 'TiltMean','TiltStd'});

            FINAL_DATA = [FINAL_DATA; row];
        end
    end
end

%% ============================================================
% 7. SAVE DATASET
%% ============================================================

csv_file = fullfile(base_path,'FINAL_DATASET_ACC.csv');
xlsx_file = fullfile(base_path,'FINAL_DATASET_ACC.xlsx');
mat_file  = fullfile(base_path,'FINAL_DATASET_ACC.mat');

if isfile(csv_file), delete(csv_file); end

writetable(FINAL_DATA, csv_file);
writetable(FINAL_DATA, xlsx_file,'FileType','spreadsheet');
save(mat_file,'FINAL_DATA');

disp('✅ DATASET SAVED: CSV + EXCEL + MAT');
