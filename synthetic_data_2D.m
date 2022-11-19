%EC 503
%Fall 2022
%Team 1
%Synthetic unbalanced data for regression

%%
rng('default')
clc
clear  all

bin_number = 10;
feature1 = [];
feature2 = [];
label = [];

r = 1.1; %dataset 2: change this value [1.1 - 2] to get different skewness | do NOT set it to exactly 1! 
Sn = 1000; 
a1 = Sn*(r-1)/(r^bin_number-1);
an = a1*r^(bin_number-1);

data_dist = a1*r.^(0:bin_number-1);  
data_dist = round(data_dist); %dataset 2: skewed 

data2 = rand(1,10);
data2 = data2/sum(data2); %normalization
data2 = data2*1000; %scaling sum to 1000

data_dist = round(data2); %uncomment this to generate the dataset 1:
%uniformly distributed random numbers


%this is to ensure that n_train = 1000
if sum(data_dist) > 1000
   data_dist(10) = data_dist(10) - (sum(data_dist) - 1000); 
end

if sum(data_dist) < 1000
    data_dist(10) = data_dist(10) - (sum(data_dist) - 1000);
end
sum(data_dist)


%% Main code for data generation in each bin
% Label space cnts between [0,10];


for n = 1:bin_number
    data_points = data_dist(n);
    
    minn = 0.01 + (n-1)/10;
    maxx = minn + 0.09; 

    for i = 1:data_points
        converged = 0;
        iteration = 0;
        while (converged == 0)
            iteration = iteration + 1; 
            fprintf('Iteration: %d\n',iteration)
            
            x1 = rand;
            x2 = rand;
            
            y_val = y_function(x1,x2)*minn; %% ADD NOISE HERE!!! in case you need

            if y_val >= minn && y_val <= maxx
            converged = 1;
            label = [label; y_val];
            feature1 = [feature1; sqrt(x1)];
            feature2 = [feature2; x2];
            end
        end
    end
        %x_measure = y_function(interval(j)) %+ normrnd(0,0.1);    
       
    
end

%figure(1)
%scatter(feature1, label, 'filled')


figure(1)
histfit(label,bin_number,'kernel')
skewness(label)

N = 250;
xvec = linspace(min(feature1), max(feature1), N);
yvec = linspace(min(feature2), max(feature2), N);
[X, Y] = ndgrid(xvec, yvec);
F = scatteredInterpolant(feature1, feature2, label);
Z = F(X, Y);
figure(2)
surf(X, Y, Z, 'edgecolor', 'none');
xlabel('feature 1')
ylabel('feature 2')
zlabel('label')
title('2D synthetic data set 1')

%s_data = [feature1 feature2 label];
%writematrix(s_data, '2D_sdata1_0.05.csv')
%% Function to calculate the feature space
function [y_value] = y_function(x1, x2)

y_value = x1.^2 + x2;

end

