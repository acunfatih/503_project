% Rec Curves

function [epsilonList,Accuracy] = plotREC(YTrain,YPred,hyp,PLOT)
    n = length(YTrain);
    if mean(YTrain) > median(YTrain)
        % Top Heavy
        t_E = prctile(YTrain,hyp.thresh);
        idx_p = YTrain >= t_E;
    else
        % Bottom Heavy (Reversed from paper)
        t_E = prctile(YTrain,100-hyp.thresh);
        idx_p = YTrain <= t_E;
    end
    
    cost = abs(YTrain-YPred);
    epsilonList = 0:.01:1;
    Accuracy = zeros(1,length(epsilonList));
    for i = 1:length(epsilonList)
        epsilon = epsilonList(i);
        idxCost = cost > epsilon;
        idx_p2 = idx_p;
        idx_p2(idxCost) = 1 - idx_p(idxCost);   
        Accuracy(i) = sum(idx_p2 == idx_p)/n;
    end
    
    if PLOT
        figure
        plot(epsilonList,Accuracy)
        xlabel('tolerance \epsilon')
        ylabel('Accurancy')
    end
end