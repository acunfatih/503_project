% Calculates error for all metrics


function [MSE, MAE, GME, CWE, BMSE, MAPE] = inference(Yhat, Y, hyp)
    MSE = calculateCost('MSE',Yhat,Y,hyp);
    MAE = calculateCost('MAE',Yhat,Y,hyp);
    GME = calculateCost('GME',Yhat,Y,hyp);
    CWE = calculateCost('CWE',Yhat,Y,hyp);
    BMSE = calculateCost('BMSE',Yhat,Y,hyp);
    MAPE = calculateCost('MAPE',Yhat,Y,hyp);
end