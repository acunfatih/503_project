% Predict Y based on model and Theta

function YPred = predictY(model,theta,XTrain)

    %Define X extended in case model uses that
    XExt = [XTrain;ones(1,size(XTrain,2))];
    switch model
        case 'LinearRegression'
            YPred = theta'*XExt;         

    end
end