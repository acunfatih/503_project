% Kernel Ridge Regression

function [K_fun,invK,K] = KRR(XTrain,hyp)
    
    switch hyp.kernel
        case 'linear'
            c = hyp.c;
            K = k_linear(XTrain, XTrain, c);
            K_fun = @k_linear;
            
        case 'RBF'
            sigma = hyp.k_sigma;
            K = k_rbf(XTrain, XTrain, sigma);
            K_fun = @k_rbf;
    end   
    
    nTrain = size(XTrain,2);
    lambda = hyp.lambda; 
    ONEn = ones(nTrain,1);
    Cn = eye(nTrain) - 1/nTrain * (ONEn * ONEn');
    
%     K = zeros(nTrain);
%     for i = 1:nTrain
%         for j = 1:nTrain
%             K(i,j) = K_fun(XTrain(:,i),XTrain(:,j));
%         end
%     end
    invK = inv(lambda * eye(nTrain) + Cn' * K * Cn);    
    
end

function dist_matrix = pairwise_dists(data, centers)
    data_squared = sum(data.^2,2);
    centers_squared = sum(centers.^2,2);
    ss = data_squared + centers_squared';
    cross_term = -2 * data * centers.';
    dist_matrix = ss + cross_term;
end

function k=k_linear(x,y,c)
    k = c + x' * y;
end

function k = k_rbf(x, y, sigma)
    dist_matrix = pairwise_dists(x',y'); % nxn dist matrix
    
    sig_term = 2 * sigma^2;
    k = exp(-1 * dist_matrix./ sig_term);
end