function cost = seraCost(YPred, YTrain)

phi_val = fitProbabilisticLoss(YTrain, 0); %relevance values

T = 1000;
%exclude the first and the last phi terms
inter_terms = 0;
%over 999 terms
for i = 0.001:0.001:0.999
    indx_vec = find(phi_val >= i);
    inter_terms = inter_terms + mean((YPred(indx_vec) - YTrain(indx_vec)).^2);
end

indx1 = find(phi_val >= 0);
term1 = 0.5*mean((YPred(indx1) - YTrain(indx1)).^2);

indx2 = find(phi_val >= 1));
term2 = 0.5*mean((YPred(indx2) - YTrain(indx2)).^2);

%summing 1001 terms
cost = (term1 + term2 + inter_terms)/T;
end