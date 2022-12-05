% Kernel Ridge Regression

function [K_fun,invK,K] = KRR(XTrain,hyp)

    switch hyp.kernel
        case 'linear'
            c = hyp.c;
            K_fun = @(u,v) (c + u'*v);
            
        case 'RBF'
            sigma = hyp.k_sigma;
            K_fun = @(u,v) exp(-((vecnorm(u-v)).^2)/(2*sigma.^2));
    end

    nTrain = size(XTrain,2);
    lambda = hyp.lambda; 
    ONEn = ones(nTrain,1);
    Cn = eye(nTrain) - 1/nTrain * (ONEn * ONEn');
    
    K = zeros(nTrain);
    for i = 1:nTrain
        for j = 1:nTrain
            K(i,j) = K_fun(XTrain(:,i),XTrain(:,j));
        end
    end
    
    invK = inv(lambda * eye(nTrain) + Cn' * K * Cn);
    
    
    
end