%% This is the main script
clear all
close all
clc

%% Configurations

% name of the dataset
dataSet = 'cali';

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
costFunction = 'CWE';

% optimizer can be moved here too but I didn't want to touch to not to 
% change the code

%% preprocess 
if preprocess
    preprocess_save_data(dataSet);
end 
%% Load dataset to use
path = strcat('data/', dataSet, '.mat');
max_label = strcat('data/max_label_', dataSet, '.mat');
load(path);
load(max_label);

%% One example using Linear regression and MSE

% Define optimizer function that will be used to find optimized theta
fun = @(theta)optimizedFunction(theta,model,costFunction,XTrain,YTrain);

% Initialize theta0 randomly. theta0 must have the correct size for the model that
% you are using.
d = size(XTrain,1);
theta0 = initializeTheta(model,d);


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

[theta,fval] = fminsearch(fun,theta0,options);
[theta2,fval2] = fminunc(fun,theta0,options);
% There are two difference search algorithms we can use. Take better of two
if fval2 < fval
    theta = theta2;
end

% Evaluate performance (can use optimizedFunction or can use other method)

cost = optimizedFunction(theta,model,costFunction,XTrain,YTrain);

% OR

% Predict YPred once
YPred = predictY(model,theta,XTrain);

% Denormalize the predictions
YPred = YPred * rangeData(end) + minData(end);
YTrain = YTrain * rangeData(end) + minData(end);

% calculate cost multiple times with different cost functions
MSE = calculateCost('MSE',YPred,YTrain)
MAE = calculateCost('MAE',YPred,YTrain)
GME = calculateCost('GME',YPred,YTrain)
CWE = calculateCost('CWE',YPred,YTrain)




