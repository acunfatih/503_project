
% Trains model and then makes predictions based on input points.


function [YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = ...
    trainAndPredict(model,costFunction,hyp,rangeData,minData,XTrainNorm,XValNorm,XTestNorm,YTrainNorm,YValNorm,YTestNorm) 

    options = optimset('Display',...
                        'iter',...
                        'PlotFcns',@optimplotfval,...
                        'MaxFunEvals',1E3,...
                        'MaxIter',1E3,...
                        'TolX',1e-4,...
                        'TolFun',1e-4);

    switch costFunction
        case 'RR'
            theta = ridge(YTrainNorm',XTrainNorm',hyp.lambda,0);
            theta = [theta(2:end);theta(1)];
            
        case 'KRR'
            [K_fun,invK,K] = KRR(XTrainNorm,hyp);
    
        otherwise
    
            % Define optimizer function that will be used to find optimized
            % theta. The variables passed are freezed as passed, so you must
            % rerun this if you update the variables.
            fun = @(theta)optimizedFunction(theta,model,costFunction,XTrainNorm,YTrainNorm,hyp);
    
            % Initialize theta0 randomly. theta0 must have the correct size for the model that
            % you are using.
            d = size(XTrainNorm,1);
            theta0 = initializeTheta(model,d);
    
            % Optimize function. There are two difference search algorithms we can use. Take better of two
            rng(1)
            [theta,fval] = fminsearch(fun,theta0,options);
            try
                [theta2,fval2] = fminunc(fun,theta0,options);
                if fval2 < fval
                    theta = theta2;
                end
            end
            
    end
    
    
    if strcmp(costFunction,'KRR')
        YPred_trainNorm = predictYKernel(XTrainNorm,YTrainNorm,XTrainNorm,K_fun,invK,K,hyp);
        YPred_valNorm = predictYKernel(XTrainNorm,YTrainNorm,XValNorm,K_fun,invK,K,hyp);
        YPred_testNorm = predictYKernel(XTrainNorm,YTrainNorm,XTestNorm,K_fun,invK,K,hyp);
    else
        YPred_trainNorm = predictY(model,theta,XTrainNorm);
        YPred_valNorm = predictY(model,theta,XValNorm);
        YPred_testNorm = predictY(model,theta,XTestNorm);
    end
    
    % Denormalize the predictions
    YPred_train = YPred_trainNorm * rangeData(end) + minData(end);
    YTrain = YTrainNorm * rangeData(end) + minData(end);
    
    YPred_val = YPred_valNorm * rangeData(end) + minData(end);
    YVal = YValNorm * rangeData(end) + minData(end);
    
    YPred_test = YPred_testNorm * rangeData(end) + minData(end);
    YTest = YTestNorm * rangeData(end) + minData(end);
    
    
end