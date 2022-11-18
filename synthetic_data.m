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

r = 1.1; %dataset 2: change this value [1.1 - 2] to get different skewness | do NOT set it to exactly 1! 
Sn = 1000; 
a1 = Sn*(r-1)/(r^bin_number-1);
an = a1*r^(bin_number-1);

data_dist = a1*r.^(0:bin_number-1);  
data_dist = round(data_dist); %dataset 2: skewed 

data2 = rand(1,10);
data2 = data2/sum(data2); %normalization
data2 = data2*1000; %scaling sum to 1000

%data_dist = round(data2); %uncomment this to generate the dataset 1:
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
    
    min = 0.1 + (n-1);
    max = min + 0.9; 
    
    interval = zeros(1, data_points);
    for i = 1:data_points
        interval(i) = min + rand*(max - min); %ensures that random number does not exceed the interval length
    end
    label = [label interval];
    
    for j = 1:data_points
        %second term is noise
        x_measure = x_function(interval(j)) + normrnd(0,0.1);    
        features = [features x_measure];
    end
   
end

figure(1)
scatter(features, label', 'filled')
xlabel('features')
ylabel('labels')
title('synthetic data set 2')

figure(2)
histfit(label,bin_number,'kernel')
skewness(label)

s_data = [features' label'];
writematrix(s_data, 's_data2.csv')
%% Function to calculate the feature space
function [x_value] = x_function(y)

x_value = sqrt(y);

end