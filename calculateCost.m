% Calculate Cost based on cost function

function cost = calculateCost(costFunction,YPred,YTrain,hyp)

    switch costFunction
        case 'MSE'
            cost = mean((YTrain-YPred).^2);
        case 'MAE'
            cost = mean(abs(YTrain-YPred));

        case 'GME'
            if mean(YTrain) > median(YTrain)
                % Top Heavy
                t_E = prctile(YTrain,hyp.thresh);
                idx_p = YTrain >= t_E;
            else
                % Bottom Heavy (Reversed from paper)
                t_E = prctile(YTrain,100-hyp.thresh);
                idx_p = YTrain <= t_E;
            end
            cost_p = mean(abs(YTrain(idx_p)-YPred(idx_p)));
            cost_n = mean(abs(YTrain(~idx_p)-YPred(~idx_p)));
            cost = sqrt(cost_p * cost_n);
        case 'CWE'
            if mean(YTrain) > median(YTrain)
                % Top Heavy
                t_E = prctile(YTrain,hyp.thresh);
                idx_p = YTrain >= t_E;
            else
                % Bottom Heavy (Reversed from paper)
                t_E = prctile(YTrain,100-hyp.thresh);
                idx_p = YTrain <= t_E;
            end
            cost_p = mean(abs(YTrain(idx_p)-YPred(idx_p)));
            cost_n = mean(abs(YTrain(~idx_p)-YPred(~idx_p)));
            w = hyp.w;
            cost = w*cost_p + (1-w)*cost_n;

        case 'MAPE'
            cost = mean(abs(YTrain-YPred)/YPred);

    end
    
end