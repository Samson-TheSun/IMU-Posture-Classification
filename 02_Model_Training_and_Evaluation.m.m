clc; clear; close all;

%% ============================================================
% FINAL MACHINE LEARNING CODE
% ACCELEROMETER-BASED POSTURE CLASSIFICATION (STERNUM SENSOR)
% ============================================================

% DESCRIPTION:
% ------------------------------------------------------------
% This program implements a machine learning framework for 
% classifying human postures using accelerometer features.
%
% The dataset (FINAL_DATASET_ACC.csv) is derived from segmented 
% 45-second windows and includes statistical features.
%
% IMPORTANT:
% ------------------------------------------------------------
% Only sternum sensor data is used to ensure:
%   - consistent orientation
%   - stable gravitational reference
%   - improved classification accuracy
%
% This avoids polarity inversion issues from belt sensor data.

%% ============================================================
% 1. LOAD DATA + FILTER
%% ============================================================

data = readtable('FINAL_DATASET_ACC.csv');

% ✅ USE ONLY STERNUM SENSOR
data = data(strcmp(data.Sensor,'sternum'), :);

data.Posture = categorical(data.Posture);
labels = categories(data.Posture);

%% ============================================================
% 2. FEATURE EXTRACTION + NORMALIZATION
%% ============================================================

X = data{:, {'MeanX','MeanY','MeanZ',...
             'StdX','StdY','StdZ',...
             'RMSX','RMSY','RMSZ'}};

Y = data.Posture;

rng(1);
X = normalize(X);

%% ============================================================
% 3. MODEL DEFINITIONS
%% ============================================================

models = {'Random Forest','KNN','Logistic','Decision Tree','Gradient Boost'};
tests  = {'Full','70/30','80/20','60/40','5-Fold CV'};

results = zeros(5,5);

%% ============================================================
% 4. FULL DATA TRAINING
%% ============================================================

RF = fitcensemble(X,Y,'Method','Bag','NumLearningCycles',150);
results(1,1) = mean(predict(RF,X)==Y)*100;

KNN = fitcknn(X,Y,'NumNeighbors',7);
results(2,1) = mean(predict(KNN,X)==Y)*100;

t = templateLinear('Learner','logistic');
LR = fitcecoc(X,Y,'Learners',t);
results(3,1) = mean(predict(LR,X)==Y)*100;

DT = fitctree(X,Y);
results(4,1) = mean(predict(DT,X)==Y)*100;

GB = fitcensemble(X,Y,'Method','AdaBoostM2','NumLearningCycles',150);
results(5,1) = mean(predict(GB,X)==Y)*100;

%% ============================================================
% 5. HOLDOUT VALIDATION
%% ============================================================

splits = [0.3 0.2 0.4];

for s = 1:3
    cv = cvpartition(Y,'HoldOut',splits(s));

    Xtrain = X(training(cv),:);
    Ytrain = Y(training(cv));
    Xtest  = X(test(cv),:);
    Ytest  = Y(test(cv));

    RF = fitcensemble(Xtrain,Ytrain,'Method','Bag','NumLearningCycles',150);
    results(1,s+1) = mean(predict(RF,Xtest)==Ytest)*100;

    KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',7);
    results(2,s+1) = mean(predict(KNN,Xtest)==Ytest)*100;

    t = templateLinear('Learner','logistic');
    LR = fitcecoc(Xtrain,Ytrain,'Learners',t);
    results(3,s+1) = mean(predict(LR,Xtest)==Ytest)*100;

    DT = fitctree(Xtrain,Ytrain);
    results(4,s+1) = mean(predict(DT,Xtest)==Ytest)*100;

    GB = fitcensemble(Xtrain,Ytrain,'Method','AdaBoostM2','NumLearningCycles',150);
    results(5,s+1) = mean(predict(GB,Xtest)==Ytest)*100;
end

%% ============================================================
% 6. CROSS VALIDATION + CONFUSION MATRICES
%% ============================================================

cv = cvpartition(Y,'KFold',5);

conf_RF = zeros(numel(labels));
conf_KNN = zeros(numel(labels));
conf_LR = zeros(numel(labels));
conf_DT = zeros(numel(labels));
conf_GB = zeros(numel(labels));

acc_RF = 0; acc_KNN = 0; acc_LR = 0; acc_DT = 0; acc_GB = 0;

for i = 1:5

    Xtrain = X(training(cv,i),:);
    Ytrain = Y(training(cv,i));
    Xtest  = X(test(cv,i),:);
    Ytest  = Y(test(cv,i));

    % Random Forest
    RF = fitcensemble(Xtrain,Ytrain,'Method','Bag','NumLearningCycles',150);
    p = predict(RF,Xtest);
    acc_RF = acc_RF + mean(p==Ytest);
    conf_RF = conf_RF + confusionmat(Ytest,p);

    % KNN
    KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',7);
    p = predict(KNN,Xtest);
    acc_KNN = acc_KNN + mean(p==Ytest);
    conf_KNN = conf_KNN + confusionmat(Ytest,p);

    % Logistic
    t = templateLinear('Learner','logistic');
    LR = fitcecoc(Xtrain,Ytrain,'Learners',t);
    p = predict(LR,Xtest);
    acc_LR = acc_LR + mean(p==Ytest);
    conf_LR = conf_LR + confusionmat(Ytest,p);

    % Decision Tree
    DT = fitctree(Xtrain,Ytrain);
    p = predict(DT,Xtest);
    acc_DT = acc_DT + mean(p==Ytest);
    conf_DT = conf_DT + confusionmat(Ytest,p);

    % Gradient Boost
    GB = fitcensemble(Xtrain,Ytrain,'Method','AdaBoostM2','NumLearningCycles',150);
    p = predict(GB,Xtest);
    acc_GB = acc_GB + mean(p==Ytest);
    conf_GB = conf_GB + confusionmat(Ytest,p);
end

results(:,5) = [acc_RF acc_KNN acc_LR acc_DT acc_GB]'/5*100;

%% ============================================================
% 7. PRINT RESULTS
%% ============================================================

disp('FINAL ACCURACY (%)')
disp(array2table(results,'VariableNames',tests,'RowNames',models))

%% ============================================================
% 8. INDIVIDUAL MODEL PERFORMANCE PLOTS
%% ============================================================

for m = 1:length(models)

    figure;
    vals = results(m,:);

    bar(vals)
    ylim([0 100])
    set(gca,'XTickLabel',tests)

    ylabel('Accuracy (%)')
    title(models{m})
    grid on

    for i = 1:length(vals)
        text(i, vals(i)+2, sprintf('%.1f%%', vals(i)), ...
            'HorizontalAlignment','center','FontWeight','bold');
    end

    saveas(gcf, sprintf('%s_Performance.png',models{m}));
end

%% ============================================================
% 9. GLOBAL COMPARISON PLOT ✅ (IMPORTANT)
%% ============================================================

figure;
b = bar(results');

set(gca,'XTickLabel',tests)
legend(models,'Location','northoutside')
ylabel('Accuracy (%)')

title('Comparison of Classification Accuracy Across Models and Evaluation Strategies')
grid on

% Label values
for i = 1:size(results,1)
    x = b(i).XEndPoints;
    y = b(i).YEndPoints;

    for j = 1:length(x)
        text(x(j), y(j)+2, sprintf('%.1f%%', y(j)), ...
            'HorizontalAlignment','center','FontSize',8);
    end
end

saveas(gcf,'Accuracy_Comparison_All_Models.png')

%% ============================================================
% 10. CONFUSION MATRICES
%% ============================================================

function plot_cm(conf, labels, name)

    accuracy = sum(diag(conf)) / sum(conf(:)) * 100;

    figure;
    cm = confusionchart(conf, labels);

    cm.Title = sprintf('%s\nAccuracy = %.2f%%', name, accuracy);
    cm.XLabel = 'Predicted Class';
    cm.YLabel = 'True Class';
    cm.Normalization = 'row-normalized';
end

plot_cm(conf_RF, labels, 'Random Forest')
plot_cm(conf_KNN, labels, 'KNN')
plot_cm(conf_LR, labels, 'Logistic Regression')
plot_cm(conf_DT, labels, 'Decision Tree')
plot_cm(conf_GB, labels, 'Gradient Boosting')

disp('✅ ML analysis complete (sternum only)');