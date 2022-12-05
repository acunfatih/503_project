% Predicts values of Kernel Ridge Regression

function YPred = predictYKernel(XTrain,YTrain,XTest,K_fun,invK,K)
    nTrain = size(XTrain,2);
    nPred = size(XTest,2);
    YPred = zeros(1,nPred);
    ONEn = ones(nTrain,1);
    Cn = eye(nTrain) - 1/nTrain * (ONEn * ONEn');
      
    Ktest = zeros(nTrain,1);
    for j = 1:nPred
        for i = 1:nTrain
            Ktest(i) = K_fun(XTrain(:,i),XTest(:,j));
        end
        YPred(j) = YTrain * Cn * invK * Cn' * (Ktest - 1/nTrain*K*ONEn) + 1/nTrain * YTrain * ONEn;
        
    end

end