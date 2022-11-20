%% This is the main script
clear all
close all
clc

%% Configurations

% name of the dataset
dataSet = 'synthetic';

rList = [1:.1:2];
for i = 1:length(rList)
    r = rList(i);

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
    CFList = {'MSE','MAE','GME','CWE'};
    for j = 1:4
        costFunction = CFList{j};

        % optimizer can be moved here too but I didn't want to touch to not to 
        % change the code

        %% preprocess 
        if preprocess
            preprocess_save_data(dataSet);
        end 
        %% Load dataset to use
        if strcmp(dataSet,'synthetic')
            path = sprintf('data/2D_sdata1_r%.2f.csv',r);
            X = readmatrix(path);
            XTrain = X(:,1:2)';
            YTrain = X(:,end)';
            rangeData = 1;
            minData = 0;
        else
            path = strcat('data/', dataSet, '.mat');
            max_label = strcat('data/max_label_', dataSet, '.mat');
            load(max_label);
            load(path);
        end



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
        MSE(i,j) = calculateCost('MSE',YPred,YTrain);
        MAE(i,j) = calculateCost('MAE',YPred,YTrain);
        GME(i,j) = calculateCost('GME',YPred,YTrain);
        CWE(i,j) = calculateCost('CWE',YPred,YTrain);
    end
end



%% Plot results
results = {sqrt(MSE),MAE,GME,CWE};

% Plot r for each of the cost functions
figure
set(gcf,'color','white')
CFList2 = CFList;
CFList2{1} = 'RMSE';
for j = 1:4
    subplot(2,2,j)
    scatter(rList,results{j}(:,j),'Filled')
    title(CFList2{j})
    xlabel('r (skewness)')
    ylabel('Cost')
    ylim([0.05,.1])
end

% Plot r for full spectrum of MAE

figure
set(gcf,'color','white')
CFList2 = CFList;
CFList2{1} = 'RMSE';
for j = 1:4
    subplot(2,2,j)
    for k = 1:4
        hold on
        plot(rList,results{j}(:,k),'-')
    end       
    title(CFList2{j})
    xlabel('r (skewness)')
    ylabel(CFList2{j})
    ylim([0.05,.12])
    legend(CFList)
end






