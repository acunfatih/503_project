%% This is the main script
clear all
close all
clc

%% Configurations

% name of the dataset
dataSet = 'synthetic';
r_value = 1.1; %we need this value for calling the preprocess_save_Data

% preprocess the existing data and save to .mat, takes values 0 or 1
preprocess = 1; 

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
costFunction = 'KRR';


% hyperparameters for cost functions. These are variables that will not be
% optimized by the optimizer, but may be necessary to change.
hyp.w = .5;
hyp.thresh = 90; %Percentile
hyp.sigma = .1;
hyp.lambda = 5e-3;
hyp.k_sigma = .1; %Must be > 0.
hyp.c = 0; % Must be >= 0.
hyp.kernel = 'linear'; % Options: linear or RBF


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
path = strcat('data/', dataSet, '.mat');
load(path);


%% One example using Linear regression and MSE

switch costFunction
    case 'RR'
        theta = ridge(YTrain',XTrain',hyp.lambda,0);
        
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
        [theta,fval] = fminsearch(fun,theta0,options);
        [theta2,fval2] = fminunc(fun,theta0,options);
        if fval2 < fval
            theta = theta2;
        end
end

% Predict YPred 
if strcmp(costFunction,'KRR')
    YPred = predictYKernel(XTrain,YTrain,XTrain,K_fun,invK,K);
else
    YPred = predictY(model,theta,XTrain);
end

% Denormalize the predictions
YPred = YPred * rangeData(end) + minData(end);
YTrain = YTrain * rangeData(end) + minData(end);

% calculate cost multiple times with different cost functions
MSE = calculateCost('MSE',YPred,YTrain,hyp)
MAE = calculateCost('MAE',YPred,YTrain,hyp)
GME = calculateCost('GME',YPred,YTrain,hyp)
CWE = calculateCost('CWE',YPred,YTrain,hyp)
BMSE = calculateCost('BMSE',YPred,YTrain,hyp)
MAPE = calculateCost('MAPE',YPred,YTrain,hyp)


