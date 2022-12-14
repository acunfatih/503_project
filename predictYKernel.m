% Predicts values of Kernel Ridge Regression

function YPred = predictYKernel(XTrain,YTrain,XTest,K_fun,invK,K,hyp)
    nTrain = size(XTrain,2);
    nPred = size(XTest,2);
    YPred = zeros(1,nPred);
    ONEn = ones(nTrain,1);
    Cn = eye(nTrain) - 1/nTrain * (ONEn * ONEn');
    
    switch hyp.kernel
        case 'linear'
            c = hyp.c;
            K_fun = @(u,v) (c + u'*v);
        case 'RBF'
            sigma = hyp.k_sigma;
            K_fun = @(u,v) exp(-((vecnorm(u-v)).^2)/(2*sigma.^2));
    end   
    
    Ktest = zeros(nTrain,1);
    
    C1 = YTrain * Cn * invK * Cn';
    C2 =  1/nTrain * YTrain * ONEn;
    C3 = 1/nTrain * K * ONEn;

    for j = 1:nPred
        for i = 1:nTrain
            Ktest(i) = K_fun(XTrain(:,i),XTest(:,j));
        end
        YPred(j) = C1 * (Ktest - C3) +C2;
        
    end

end