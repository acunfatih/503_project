%% This is the main script
clear all
close all
clc


%% Load Datasets
dataSet = 'cali';
switch dataSet
    case 'bodyfat'
        load bodyfat_dataset
        XTrain = bodyfatInputs;
        YTrain = bodyfatTargets;
    case 'cali'
        [XTrain, YTrain, Xval, Yval] = load_dataset(dataSet);
end

%% One example using Linear regression and MSE

% Select model and cost function
% Current Model Options
% 1. LinearRegression
model = 'LinearRegression';

% Current CostFunction Options
% 1. MSE (Mean Square Error)
% 2. MAE (Mean Absolute Error)
costFunction = 'MAE';

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
% calculate cost multiple times with different cost functions
MSE = calculateCost('MSE',YPred,YTrain)
MAE = calculateCost('MAE',YPred,YTrain)







