%% ============================================================
%   BANK MARKETING PROJECT (Improved Version)
%   Loads data from UCI, preprocesses, trains ML models.
% ============================================================

clc; clear; close all;

%% 1. Download the dataset ZIP from UCI
disp("Downloading dataset from UCI...");

zipUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip';
zipFile = 'bank-additional.zip';
websave(zipFile, zipUrl);

disp("Download complete.");

%% 2. Unzip into workspace
unzipFolder = 'bank-data';
unzip(zipFile, unzipFolder);
disp("Unzip complete.");

%% 3. Load CSV file
csvFile = fullfile(unzipFolder, 'bank-additional', 'bank-additional-full.csv');
opts = detectImportOptions(csvFile, 'Delimiter',';');
data = readtable(csvFile, opts);

disp("Dataset loaded:");
disp(size(data));

%% 4. Encode categorical variables (BUT NOT 'y'!)
varNames = data.Properties.VariableNames;

for i = 1:numel(varNames)
    if iscellstr(data.(varNames{i})) || isstring(data.(varNames{i}))
        if ~strcmp(varNames{i}, 'y')
            data.(varNames{i}) = grp2idx(categorical(data.(varNames{i})));
        end
    end
end

%% 5. Fix binary label encoding (correct way)
data.y = categorical(data.y);
data.y = double(data.y == 'yes');   % yes=1, no=0

%% 6. Split predictors & target
X = data(:,1:end-1);
y = data.y;

%% 7. Train-test split (70/30)
rng(1);
cv = cvpartition(y, 'HoldOut', 0.3);

Xtrain = X(training(cv),:);
ytrain = y(training(cv));

Xtest = X(test(cv),:);
ytest = y(test(cv));

%% ============================================================
%   MODEL 1 — Balanced Logistic Regression
% ============================================================

disp("Training BALANCED logistic regression...");

% Weight yes=1 cases more because dataset is imbalanced
weights = ones(size(ytrain));
weights(ytrain == 1) = 10;  % increase minority class weight

mdl_log = fitclinear( ...
    Xtrain, ytrain, ...
    'Learner','logistic', ...
    'ClassNames', [0 1], ...
    'Weights', weights);

ypred_log = predict(mdl_log, Xtest);
prob_log = mdl_log.predict(Xtest);

acc_log = mean(ypred_log == ytest);
fprintf("\nLogistic Regression Accuracy: %.2f%%\n", acc_log*100);

disp("Confusion Matrix — Logistic Regression:");
disp(confusionmat(ytest, ypred_log));

%% ============================================================
%   MODEL 2 — Decision Tree
% ============================================================

disp("Training Decision Tree...");

mdl_tree = fitctree(Xtrain, ytrain);
ypred_tree = predict(mdl_tree, Xtest);

acc_tree = mean(ypred_tree == ytest);
fprintf("\nDecision Tree Accuracy: %.2f%%\n", acc_tree*100);

disp("Confusion Matrix — Decision Tree:");
disp(confusionmat(ytest, ypred_tree));

%% ============================================================
%   MODEL 3 — Random Forest (TreeBagger)
% ============================================================

disp("Training Random Forest (100 trees)...");

mdl_rf = TreeBagger(100, Xtrain, ytrain, ...
    'Method','classification', ...
    'OOBPrediction','On', ...
    'OOBPredictorImportance','On');

ypred_rf = str2double(predict(mdl_rf, Xtest));

acc_rf = mean(ypred_rf == ytest);
fprintf("\nRandom Forest Accuracy: %.2f%%\n", acc_rf*100);

disp("Confusion Matrix — Random Forest:");
disp(confusionmat(ytest, ypred_rf));

%% ============================================================
%   ROC Curves + AUC for all models
% ============================================================

figure; hold on;

% Logistic regression
[~,scores_log] = predict(mdl_log, Xtest);
[Xlog, Ylog, ~, AUClog] = perfcurve(ytest, scores_log(:,2), 1);

% Decision tree
[~,scores_tree] = resubPredict(fitctree(Xtrain,ytrain));
[Xtree, Ytree] = perfcurve(ytest, ypred_tree, 1);

% Random Forest
[~,scores_rf] = oobPredict(mdl_rf);
[Xrf, Yrf, ~, AUCrf] = perfcurve(ytrain, scores_rf(:,2), '1');

plot(Xlog,  Ylog, 'LineWidth',2);
plot(Xtree, Ytree,'LineWidth',2);
plot(Xrf,   Yrf,  'LineWidth',2);

title('ROC Curves');
legend( ...
    sprintf("Logistic (AUC = %.2f)", AUClog), ...
    "Decision Tree", ...
    sprintf("Random Forest (AUC = %.2f)", AUCrf), ...
    'Location','Best');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
grid on;

%% ============================================================
%   Feature Importance (Random Forest)
% ============================================================

figure;
bar(mdl_rf.OOBPermutedPredictorDeltaError);
title('Feature Importance (Random Forest)');
xticklabels(Xtrain.Properties.VariableNames);
xtickangle(45);
ylabel('Importance Score');
grid on;
