function cost = seraCost(YPred, YTrain)

a = (mean(YTrain)/std(YTrain))^2;
b = (std(YTrain))^2/(mean(YTrain));
phi_val = gampdf(YTrain, a); %relevance values
phi_val = (phi_val - min(phi_val))/range(phi_val);
phi_val = 1-phi_val;


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

function [norm_data,minData,rangeData] = norm_zero2one(data)
    minData = min(data);
    rangeData = range(data);
    norm_data  = (data - minData)./rangeData;
end

end