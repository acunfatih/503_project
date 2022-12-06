%EC 503
%Fall 2022
%Team 1
%Synthetic unbalanced data for regression

%%
clear all
close all
clc

rng('default')

bin_number = 10;
data_x1 = [];
data_x2 = [];
data_y = [];

npts_vec = [700; 150; 150];
r = 1.7 %dataset 2: change this value [1 - 1.7] to get different skewness 

for k = 1:3
npts = npts_vec(k);

feature1 = [];
feature2 = [];
label = [];

if r == 1 %equal distribution
    data_dist = ones(1,10) * round(npts/bin_number);
else 
    a1 = npts*(r-1)/(r^bin_number-1); %first term in geometric series
    an = a1*r^(bin_number-1); %last term in the series

    data_dist = a1*r.^(0:bin_number-1);  
    data_dist = round(data_dist); %dataset 2: skewed 
end

data_randm = rand(1,10);
data_randm = data_randm/sum(data_randm); %normalization to add up to 1
data_randm = data_randm*1000; %scaling sum to 1000

% data_dist = round(data2); %uncomment this to generate the dataset 1:
%uniformly distributed random numbers


% Flip so that high is most uncommon
data_dist = flip(data_dist);

%this is to ensure that n_train = 1000
if sum(data_dist) > npts
   data_dist(1) = data_dist(1) - (sum(data_dist) - npts); 
end

if sum(data_dist) < npts
    data_dist(1) = data_dist(1) - (sum(data_dist) - npts);
end
sum(data_dist)


%% Main code for data generation in each bin
% Label space cnts between [0,2];

%for each of ten bins
for n = 1:bin_number
    data_points = data_dist(n);
    
    %Interval lengh for each bin label space is 0.2
    minn = (n-1)/10 * 2;
    maxx = minn + 0.1 * 2; 

    %for each point in each of 10 bins
    for i = 1:data_points
        converged = 0;
        iteration = 0;
        while (converged == 0)
            iteration = iteration + 1; 
            fprintf('Iteration: %d\n',iteration)
            
            x1 = rand;
            x2 = rand;
            
            y_val = y_function(x1,x2);
            y_val = y_val +  normrnd(0,0.05); % Add noise to measurement

            if y_val > minn && y_val <= maxx
                converged = 1;
                label = [label; y_val];
                feature1 = [feature1; x1];
                feature2 = [feature2; x2];
            end
        end
    end
    
end

data_x1  = [data_x1; feature1];
data_x2  = [data_x2; feature2];
data_y  = [data_y; label];

end


%% Plot TRAINING data for visualization
% calculate grid to plot sampled data
N = 250;
xvec = linspace(min(feature1), max(feature1), N);
yvec = linspace(min(feature2), max(feature2), N);
[X, Y] = ndgrid(xvec, yvec);
F = scatteredInterpolant(feature1, feature2, label);
Z = F(X, Y);

% calculate grid to plot sampled data
Z_truth = y_function(X,Y);


% plot histogram
figure(1)
histogram(label,bin_number)
set(gcf,'color','white')
title(sprintf('2D synthetic data set: r = %.2f',r))
skewv = skewness(label);
fprintf('Skewness: %d\n',skewv)

% Plot samples data
figure(2)
surf(X, Y, Z, 'edgecolor', 'none');
xlabel('feature 1')
ylabel('feature 2')
zlabel('label')
title(sprintf('2D synthetic data set: r = %.2f',r))
set(gcf,'color','white')

figure(3)
subplot(1,2,1)
contourf(X,Y,Z)
xlabel('feature 1')
ylabel('feature 2')
zlabel('label')
colorbar
title(sprintf('synthetic dataset: r = %.2f',r'))
set(gcf,'color','white')

% Plot underlying function 
subplot(1,2,2)
contourf(X,Y,Z_truth)
xlabel('feature 1')
ylabel('feature 2')
zlabel('label')
colorbar
title('Underlying Function')
set(gcf,'color','white')

% Plot underlying function with sample points
figure(4)
contour(X, Y, Z_truth);
hold on
scatter(feature1, feature2,15,label,'Filled')
xlabel('feature 1')
ylabel('feature 2')
zlabel('label')
colorbar
title(sprintf('2D synthetic data set: r = %.2f',r))
set(gcf,'color','white')


s_data = [data_x1 data_x2 data_y];
writematrix(s_data, sprintf('data/2D_sdata1_r%.2f.csv',r))

%% Underlying function
function [y_value] = y_function(x1, x2)

y_value = x1.^2 + x2;

end

