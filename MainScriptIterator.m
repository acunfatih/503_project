%% This is the main script
clear all
close all
clc

models = ["LinearRegression"];
costFunctions = ["MSE" 
    "MAE" 
    "GME" 
    "CWE" 
    "BMSE" 
    "RR" 
    "KRR" 
    "PLOSS"];

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
    hyp.lambda = 5e-3; %Used by RR and KRR
    hyp.kernel = 'RBF'; % Options: linear or RBF
    hyp.k_sigma = .1; %Must be > 0. Used by KRR-RBF
    hyp.c = 0; % Must be >= 0. Used by KRR-linear
    
    
    
    % Optimize theta (Note, if the model and cost function you are using has a
    % closed form solution, you can use that. But this seems rare since we are
    % mixing and matching models and cost functions).
    options = optimset('Display',...
                        'iter',...
                        'PlotFcns',@optimplotfval,...
                        'MaxFunEvals',1E5,...
                        'MaxIter',1E5,...
                        'TolX',1e-4,...
                        'TolFun',1e-4);
    
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
    load(path);
    
    %%
    if strcmp(costFunction, 'PLOSS')
        hyp.phi_y = fitProbabilisticLoss(YTrain, 0);
    end
    
    %% One example using Linear regression and MSE
    
    switch costFunction
        case 'RR'
            theta = ridge(YTrain',XTrain',hyp.lambda,0);
            theta = [theta(2:end);theta(1)];
            
        case 'KRR'
            [K_fun,invK,K] = KRR(XTrain,hyp);
    
        otherwise
    
            % Define optimizer function that will be used to find optimized
            % theta. The variables passed are freezed as passed, so you must
            % rerun this if you update the variables.
            fun = @(theta)optimizedFunction(theta,model,costFunction,XTrain,YTrain,hyp);
    
            % Initialize theta0 randomly. theta0 must have the correct size for the model that
            % you are using.
            d = size(XTrain,1);
            theta0 = initializeTheta(model,d);
    
            % Optimize function. There are two difference search algorithms we can use. Take better of two
            rng(1)
            [theta,fval] = fminsearch(fun,theta0,options);
            [theta2,fval2] = fminunc(fun,theta0,options);
            if fval2 < fval
                theta = theta2;
            end
    end
    
    
    if strcmp(costFunction,'KRR')
        YPred = predictYKernel(XTrain,YTrain,XTrain,K_fun,invK,K);
        YPred_val = predictYKernel(XVal,YVal,XVal,K_fun,invK,K);
        YPred_test = predictYKernel(XTest,YTest,XTest,K_fun,invK,K);
    else
        YPred = predictY(model,theta,XTrain);
        YPred_val = predictY(model,theta,XVal);
        YPred_test = predictY(model,theta,XTest);
    end
    
    % Denormalize the predictions
    YPred = YPred * rangeData(end) + minData(end);
    YTrain = YTrain * rangeData(end) + minData(end);
    
    YPred_val = YPred_val * rangeData(end) + minData(end);
    YVal = YVal * rangeData(end) + minData(end);
    
    YPred_test = YPred_test * rangeData(end) + minData(end);
    YTest = YTest * rangeData(end) + minData(end);
    % calculate cost multiple times with different cost functions
    
    [MSE, MAE, GME, CWE, BMSE, MAPE] = inference(YPred, YTrain, hyp);
    [MSE_val, MAE_val, GME_val, CWE_val, BMSE_val, MAPE_val] = inference(YPred_val, YVal, hyp);
    [MSE_test, MAE_test, GME_test, CWE_test, BMSE_test, MAPE_test] = inference(YPred_test, YTest, hyp);

    dataStr = dataSet;
    
    if strcmp(dataSet, "synthetic")
        dataStr = strcat(dataSet,'_r=',num2str(r_value));
    end
    path = strcat('results/', model,'_', dataStr, '_', costFunction);

    mkdir(path);
    plotParity(YTrain,YPred,strcat(path,'/parity'));
    [epsilonList,Accuracy] = plotREC(YTrain,YPred,hyp,1,strcat(path,'/REC'));
    
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

function [MSE, MAE, GME, CWE, BMSE, MAPE] = inference(Yhat, Y, hyp)
    MSE = calculateCost('MSE',Yhat,Y,hyp);
    MAE = calculateCost('MAE',Yhat,Y,hyp);
    GME = calculateCost('GME',Yhat,Y,hyp);
    CWE = calculateCost('CWE',Yhat,Y,hyp);
    BMSE = calculateCost('BMSE',Yhat,Y,hyp);
    MAPE = calculateCost('MAPE',Yhat,Y,hyp);
end


