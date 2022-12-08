%% This is the main script
clear all
close all
clc

models = ["LinearRegression"];
costFunctions = [
    "KRR"];

r_values = [1:0.1:1.7]';
dataSets = ["synthetic"
    "cali"];


%%
for m = 1:size(models,1)
    for c = 1:size(costFunctions,1)
        for d = 1:size(dataSets,1)
            if strcmp(dataSets(d), "synthetic")
                for r = 1:size(r_values,1)
                    fprintf("Training model: %s, costFunction: %s, data: %s, r_value: %f \n", models(m), costFunctions(c), dataSets(d), r_values(r));
                    train_eval(r_values(r), dataSets(d), models(m), costFunctions(c));
                end
            else
                fprintf("Training model: %s, costFunction: %s, data: %s \n", models(m), costFunctions(c), dataSets(d));
%                 train_eval(r_value, dataSet, model, costFunction);
            end
            
        end
    end
end

%%

function train_eval(r_value, dataSet, model, costFunction)
    % Configurations
    
    % preprocess the existing data and save to .mat, takes values 0 or 1
    preprocess = 0; 

    % hyperparameters for cost functions. These are variables that will not be
    % optimized by the optimizer, but may be necessary to change.
    hyp.w = .5; %Used by CWE
    hyp.thresh = 90; %Percentile; Used by CWE,GME
    hyp.sigma = .1; %Used by BMSE
    hyp.lambda = .01; %Used by RR and KRR
    hyp.kernel = 'RBF'; % Options: linear or RBF
    hyp.k_sigma = 1; %Must be > 0. Used by KRR-RBF
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

    
    %%
    if strcmp(costFunction, 'PLOSS')
        hyp.phi_y = fitProbabilisticLoss(YTrain, 0);
    end
    
    %% Train and Predict
    [YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = ...
        trainAndPredict(model,costFunction,hyp,rangeData,minData,XTrain,XVal,XTest,YTrain,YVal,YTest);
    
    % calculate cost multiple times with different cost functions
    
    [MSE, MAE, GME, CWE, BMSE, MAPE] = inference(YPred_train, YTrain, hyp);
    [MSE_val, MAE_val, GME_val, CWE_val, BMSE_val, MAPE_val] = inference(YPred_val, YVal, hyp);
    [MSE_test, MAE_test, GME_test, CWE_test, BMSE_test, MAPE_test] = inference(YPred_test, YTest, hyp);

    dataStr = dataSet;
    
    if strcmp(dataSet, "synthetic")
        dataStr = strcat(dataSet,'_r=',num2str(r_value));
    end
    path = strcat('results/', model,'_', dataStr, '_', costFunction);

    mkdir(path);
    plotParity(YTrain,YPred_train,strcat(path,'/parity'),0);
    [epsilonList,Accuracy] = plotREC(YTrain,YPred_train,hyp,0,strcat(path,'/REC'));
    
    %%
    if ~isfile('results/results.csv')
        fid = fopen( 'results/results.csv', 'w' );
        fprintf(fid, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 'model', 'dataStr', ...
        'costFunction', 'MSE', 'MAE', 'GME', 'CWE', 'BMSE', 'MAPE', ...
        'MSE_val', 'MAE_val', 'GME_val', 'CWE_val', 'BMSE_val', ...
        'MAPE_val', 'MSE_test', 'MAE_test', 'GME_test', 'CWE_test', 'BMSE_test', 'MAPE_test', 'date');
    else
        fid = fopen( 'results/results.csv', 'a+' );
    end
    fprintf( fid, '%s,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s\n', model, dataStr, ...
        costFunction, MSE, MAE, GME, CWE, BMSE, MAPE, ...
        MSE_val, MAE_val, GME_val, CWE_val, BMSE_val, ...
        MAPE_val, MSE_test, MAE_test, GME_test, CWE_test, BMSE_test, MAPE_test, datestr(now,'DD HH:MM:SS'));
    fclose( fid );
end




