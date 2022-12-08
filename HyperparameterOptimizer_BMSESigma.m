%% This is the main script
clear all
close all
clc

% name of the dataset
r_value = 1.1; %we need this value for calling the preprocess_save_Data
% dataSet = 'cali';
dataSet = 'synthetic';

% preprocess the existing data and save to .mat, takes values 0 or 1
preprocess = 0; 

% Select model and cost function
% Current Model Options
% 1. LinearRegression
model = 'LinearRegression';

% Current CostFunction Options
% 1. MSE (Mean Square Error)
% 2. MAE (Mean Absolute Error)
% 3. GME (Geometric Mean Error)
% 4. CWE (Class Weighted Error)
% 5. BMSE (Balanced MSE)
% 6. MAPE (Mean Absolute Percentage Error)
% 7. RR (Ridge Regression)
% 8. KRR (Kernel Ridge Regression: Very Slow)
% 9. PLOSS (Probabilistic Loss)
% 10. SERA (SERA)
costFunction = 'BMSE';


% hyperparameters for cost functions. These are variables that will not be
% optimized by the optimizer, but may be necessary to change.
hyp.w = .9; %Used by CWE
hyp.thresh = 90; %Percentile; Used by CWE,GME
hyp.sigma = .1; %Used by BMSE
hyp.lambda = 5e-3; %Used by RR and KRR
hyp.kernel = 'RBF'; % Options: linear or RBF
hyp.k_sigma = .1; %Must be > 0. Used by KRR-RBF
hyp.c = 0; % Must be >= 0. Used by KRR-linear




%% preprocess 
if preprocess
    preprocess_save_data(dataSet,r_value);
end 
%% Load dataset to use
if strcmp(dataSet,'synthetic')
    path = sprintf('data/synthetic%.1f.mat',r_value);
else
    path = strcat('data/', dataSet, '.mat');
end
load(path,'minData','rangeData','XTest','XTrain','XVal','YTest','YTrain','YVal');
YTrainOrig = YTrain;
YValOrig = YVal;
YTestOrig = YTest;


sigmaList = logspace(-2,3,11);

for i = 1:length(sigmaList)
    hyp.sigma = sigmaList(i);
    
    [YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = ...
    trainAndPredict(model,costFunction,hyp,rangeData,minData,XTrain,XVal,XTest,YTrainOrig,YValOrig,YTestOrig);

    [MSE_train(i), MAE_train(i), GME_train(i), CWE_train(i), BMSE_train(i), MAPE_train(i)] = inference(YPred_train, YTrain, hyp);
    [MSE_val(i), MAE_val(i), GME_val(i), CWE_val(i), BMSE_val(i), MAPE_val(i)] = inference(YPred_val, YVal, hyp);
    [MSE_test(i), MAE_test(i), GME_test(i), CWE_test(i), BMSE_test(i), MAPE_test(i)] = inference(YPred_test, YTest, hyp);


    path = sprintf('hypresults/%s_%.1f_%s_%.2e',dataSet,r_value,costFunction,hyp.sigma);
    mkdir(path);
    plotParity(YPred_val,YVal,strcat(path,'/parity'),1);
    [epsilonList,Accuracy] = plotREC(YPred_val,YVal,hyp,1,strcat(path,'/REC'));

end

%% Comparison Figure

figure
displaynames = {'MSE','GME','MAE'};
for i = 1:3
    switch i
        case 1
            vals = MSE_train;
            colorVal = 'r';
        case 2
            vals = GME_train;
            colorVal = 'g';
        case 3
            vals = MAE_train;
            colorVal = 'b';
    end
    
    plot(sigmaList(2:end-1),vals(2:end-1),colorVal,'DisplayName',displaynames{i})
    hold on
    [minVal,idx] = min(vals);
    scatter(sigmaList(idx),minVal,colorVal,'Filled','DisplayName','Min of Function')
end
legend('Location','Best')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
xlabel('Sigma')
ylabel('Error')
title('Validation Error')

figure
for i = 1:3
    switch i
        case 1
            vals = MSE_train;
            colorVal = 'r';
        case 2
            vals = GME_train;
            colorVal = 'g';
        case 3
            vals = MAE_train;
            colorVal = 'b';
    end
    
    plot(sigmaList(2:end-1),vals(2:end-1),colorVal,'DisplayName',displaynames{i})
    hold on
    [minVal,idx] = min(vals);
    scatter(sigmaList(idx),minVal,colorVal,'Filled','DisplayName','Min of Function')
end
legend('Location','Best')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
xlabel('Sigma')
ylabel('Error')
title('Training Error')

%% 
% 
% fid = fopen( 'hypresults/results.csv', 'a+' );
% 
% 
% fprintf( fid, '%s,%s,%s,%f,%f,%f,%f,%f,%f,%s\n', model, dataStr, ...
%     costFunction, MSE, MAE, GME, CWE, BMSE, MAPE, datestr(now,'DD HH:MM:SS'));
% fclose( fid );


%% 

%%
