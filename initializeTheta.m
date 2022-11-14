% This function initializes theta with the proper dimensions for the model
% being used. It uses normrnd to generate random initializations

function theta0 = initializeTheta(model,d)

    switch model
        case 'LinearRegression'
            theta0 = normrnd(0,1,d+1,1);
    end

end