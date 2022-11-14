% Calculate Cost based on cost function

function cost = calculateCost(costFunction,YPred,YTrain)

    switch costFunction
        case 'MSE'
            cost = mean((YTrain-YPred).^2);
        case 'MAE'
            cost = mean(abs(YTrain-YPred));
    end
    
end