%EC 503
%Fall 2022
%Team 1
%Synthetic unbalanced data for regression

%%
rng('default')
clc
clear  all

bin_number = 10;
features = [];
label = [];

data_points_distribution = [10, 5, 15, 8, 22, 10, 4, 12, 7, 7]; %dataset 1: random
%data_points_distribution = [1:2:19]; %dataset 2: skewed 


for n = 1:bin_number
    data_points = data_points_distribution(n);
    
    min = 0.1 + (n-1);
    max = min + 0.9; 
    
    interval = zeros(1, data_points);
    for i = 1:data_points
        interval(i) = min + rand*(max - min); %ensures that random number does not exceed the interval length
    end
    label = [label interval];
    
    for j = 1:data_points
        %second term is noise
        x_measure = x_function(interval(j)) + normrnd(mean(label),std(label));    
        features = [features x_measure];
    end
   
end

figure(1)
scatter(features, label', 'filled')
xlabel('features')
ylabel('labels')
title('synthetic data set 1')

figure(2)
histfit(label,bin_number,'kernel')
skewness(label)
%%
function [x_value] = x_function(y)

x_value = sqrt(y);

end