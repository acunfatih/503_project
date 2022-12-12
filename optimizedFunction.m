%% This function is used in fminsearch and fminunc to optimize theta


function cost = optimizedFunction(theta,model,costFunction,XTrain,YTrain,hyp)

    % Predict YPred
    YPred = predictY(model,theta,XTrain);

    % calculate cost
    cost = calculateCost(costFunction,YPred,YTrain,hyp);
    

end