%% This function is used in fminsearch to optimize theta


function cost = optimizedFunction(theta,model,costFunction,XTrain,YTrain)

    % Predict YPred
    YPred = predictY(model,theta,XTrain);

    % calculate cost
    cost = calculateCost(costFunction,YPred,YTrain);
    

end