% Rec Curves

function [epsilonList,Accuracy] = plotREC(YTrain,YPred,hyp,PLOT,path)
    w = hyp.w;
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
    epsilonList = linspace(0,max(cost)*1.1);
    Accuracy = zeros(1,length(epsilonList));
    for i = 1:length(epsilonList)
        epsilon = epsilonList(i);
        
        TPR = sum(cost(idx_p) <= epsilon)/sum(idx_p);
        AccuracyTPR(i) = TPR;
        
        TNR = sum(cost(~idx_p) <= epsilon)/sum(~idx_p);
        AccuracyTNR(i) = TNR;
        
        AccuracyGM(i) = sqrt(TPR * TNR);
        
        AccuracyCWA(i) = w*TPR + (1-w) * TNR;
    end
    
    if PLOT
        fig = figure;
        plot(epsilonList,AccuracyTPR)
        hold on
        plot(epsilonList,AccuracyTNR)
        plot(epsilonList,AccuracyGM)
        plot(epsilonList,AccuracyCWA)
        legend('REC_{TPR}','REC_{TNR}','REC_{G-Mean}','REC_{CWA}')
        xlabel('tolerance \epsilon')
        ylabel('Accurancy')
        savefig(path);
        saveas(fig,[path,'.png'])
    end
end