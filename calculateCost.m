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
            
        case 'BMSE'
            sumVal = 0;
            for i = 1:length(YPred)
                sumVal = sumVal + exp(-sqrt(mean((YPred(i)-YTrain(i)).^2))/(2*hyp.sigma.^2));
            end
            cost = -log((exp(-sqrt(mean((YPred-YTrain).^2))/(2*hyp.sigma.^2))./sumVal));
        
        case 'PLOSS'
            cost = mean((YTrain-YPred).^2 + ((YTrain - YPred).^2+10^-9) .* hyp.phi_y);

    end
    
end