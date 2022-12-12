% Slide Explanations
clear all
close all
clc


% Load synthetic dataset

r_value = 1.4; 
dataSet = 'synthetic';

if strcmp(dataSet,'synthetic')
    path = sprintf('data/synthetic%.1f.mat',r_value);
else
    path = strcat('data/', dataSet, '.mat');
end
load(path);

%% Demonstrate Psuedo Class Creation

%Create scatter plot that can take a threshold

figure
swarmchart(ones(1,length(YTrain)),YTrain,36,'b','Filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5)
ylabel('Label')
xticks([])
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.7')

% Apply Threshold
t_E = prctile(YTrain,90);

figure
swarmchart(ones(1,length(YTrain)),YTrain,36,'b','Filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5)
hold on
yline(t_E,'r--','LineWidth',3)
ylabel('Label')
xticks([])
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.7')
legend('Data','90% Threshold')

% Plot as separate classes
idx = YTrain > t_E;

for i = 1:length(YTrain)
    if idx(i)
        colorList(i,:) = [0,0,1];
    else
        colorList(i,:) = [0,1,0];
    end
end

figure
swarmchart(ones(1,length(YTrain)),YTrain,36,colorList,'Filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5)
hold on
yline(t_E,'r--','LineWidth',3)
ylabel('Label')
xticks([])
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.7')


% Get Legend
figure
swarmchart(ones(1,length(YTrain(idx))),YTrain(idx),36,[0,0,1],'Filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5)
hold on
swarmchart(ones(1,length(YTrain(~idx))),YTrain(~idx),36,[0,1,0],'Filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5)
yline(t_E,'r--','LineWidth',3)
ylabel('Label')
xticks([])
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.7')
legend('Class P','Class N','90% Threshold')



%% Try two at describing Psuedo Class

%Create scatter plot that can take a threshold

figure
histogram(YTrain)
xlabel('Label')
ylabel('Number of Points')
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.4')

% CDF
t_E = prctile(YTrain,90);

figure
cdfplot(YTrain)
ylabel('Fraction of Data Below Threshold')
xlabel('Threshold')
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.4')
set(gcf,'color','white')
hold on
yline(.9,'r--')
ylabel('Label')
set(gcf,'color','white')
title('Normalized Synthetic Dataset: r=1.4')
legend('CDF','90% Threshold')



%% CWE Sweeps
model = 'LinearRegression';
costFunctions = [
    "MAE"
    "RR"
%     "GME"
%     "CWE"
    "PLOSS"
    "SERA"
    "BMSE"
    "KRR"
    ];

hyp.w = .5; %Used by CWE
hyp.thresh = 90; %Percentile; Used by CWE,GME
hyp.sigma = .1; %Used by BMSE
hyp.lambda = 5e-3; %Used by RR and KRR
hyp.kernel = 'RBF'; % Options: linear or RBF
hyp.k_sigma = .1; %Must be > 0. Used by KRR-RBF
hyp.c = 0; % Must be >= 0. Used by KRR-linear

wList = 0:.1:1;

for c = 1:length(costFunctions)
    costFunction = costFunctions(c);
    [YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = loadPreds(dataSet,r_value,model,costFunction);
    costGME(c) = calculateCost('GME',YPred_val,YVal,hyp);
    for w = 1:length(wList)
        hyp.w = wList(w);
        costCWE(c,w) = calculateCost('CWE',YPred_val,YVal,hyp);
    end
end


% Create Legend
figure
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    plot(wList,costCWE(i,:),'Color',color,'DisplayName',costFunctions{i},'LineWidth',2) 
    hold on
end
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    scatter(0.5,costGME(i),36,color,'Filled') 
    hold on
end
set(gcf,'color','white')
xlabel('w')
ylabel('Validation Error')
title('Synthetic Dataset: r = 1.7')



% Create Legend
figure
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    plot(wList,costCWE(i,:),'Color',color,'DisplayName',costFunctions{i},'LineWidth',2) 
    hold on
end
legend('Location','Best')
set(gcf,'color','white')

figure
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    scatter(0.5,costGME(i),36,color,'Filled','DisplayName',costFunctions{i}) 
    hold on
end
legend



%% Rec Curve with SERA
model = 'LinearRegression';
costFunction = 'SERA';

hyp.w = .5; %Used by CWE
hyp.thresh = 90; %Percentile; Used by CWE,GME
hyp.sigma = .1; %Used by BMSE
hyp.lambda = 5e-3; %Used by RR and KRR
hyp.kernel = 'RBF'; % Options: linear or RBF
hyp.k_sigma = .1; %Must be > 0. Used by KRR-RBF
hyp.c = 0; % Must be >= 0. Used by KRR-linear

[YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = loadPreds(dataSet,r_value,model,costFunction);

w = hyp.w;
n = length(YVal);
if mean(YVal) > median(YVal)
    % Top Heavy
    t_E = prctile(YVal,hyp.thresh);
    idx_p = YVal >= t_E;
else
    % Bottom Heavy (Reversed from paper)
    t_E = prctile(YVal,100-hyp.thresh);
    idx_p = YVal <= t_E;
end

cost = abs(YVal-YPred_val);
epsilonList = linspace(0,max(cost)*1.1);
Accuracy = zeros(1,length(epsilonList));
for i = 1:length(epsilonList)
    epsilon = epsilonList(i);

    TPR = sum(cost(idx_p) <= epsilon)/sum(idx_p);
    AccuracyTPR(i) = TPR;

    TNR = sum(cost(~idx_p) <= epsilon)/sum(~idx_p);
    AccuracyTNR(i) = TNR;

    AccuracyGME(i) = sqrt(TPR * TNR);

    AccuracyCWA(i) = w*TPR + (1-w) * TNR;
end

fig = figure;
plot(epsilonList,AccuracyTPR,'LineWidth',2)
hold on
plot(epsilonList,AccuracyTNR,'LineWidth',2)
plot(epsilonList,AccuracyGME,'LineWidth',2)
plot(epsilonList,AccuracyCWA,'LineWidth',2)
legend('REC_{TPR}','REC_{TNR}','REC_{GME}','REC_{CWA}')
xlabel('Error Tolerance \epsilon')
ylabel('Validation Accurancy')
title('Synthetic Dataset: r = 1.7')
subtitle('Training Cost Function = SERA')
set(gcf,'color','white')

epsilonListSERA = epsilonList;
AccuracyTPRSERA = AccuracyTPR;


%% Rec Curve with CWE
model = 'LinearRegression';
costFunction = 'CWE';

hyp.w = .7; %Used by CWE
hyp.thresh = 90; %Percentile; Used by CWE,GME
hyp.sigma = .1; %Used by BMSE
hyp.lambda = 5e-3; %Used by RR and KRR
hyp.kernel = 'RBF'; % Options: linear or RBF
hyp.k_sigma = .1; %Must be > 0. Used by KRR-RBF
hyp.c = 0; % Must be >= 0. Used by KRR-linear

[YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = ...
            trainAndPredict(model,costFunction,hyp,rangeData,minData,XTrain,XVal,XTest,YTrain,YVal,YTest);

w = 0.5;
n = length(YVal);
if mean(YVal) > median(YVal)
    % Top Heavy
    t_E = prctile(YVal,hyp.thresh);
    idx_p = YVal >= t_E;
else
    % Bottom Heavy (Reversed from paper)
    t_E = prctile(YVal,100-hyp.thresh);
    idx_p = YVal <= t_E;
end

cost = abs(YVal-YPred_val);
epsilonList = linspace(0,max(cost)*1.1);
Accuracy = zeros(1,length(epsilonList));
for i = 1:length(epsilonList)
    epsilon = epsilonList(i);

    TPR = sum(cost(idx_p) <= epsilon)/sum(idx_p);
    AccuracyTPR(i) = TPR;

    TNR = sum(cost(~idx_p) <= epsilon)/sum(~idx_p);
    AccuracyTNR(i) = TNR;

    AccuracyGME(i) = sqrt(TPR * TNR);

    AccuracyCWA(i) = w*TPR + (1-w) * TNR;
end


% Plot Just W
fig = figure;
plot(epsilonList,AccuracyTPR,'LineWidth',2)
hold on
plot(epsilonList,AccuracyTNR,'LineWidth',2)
plot(epsilonList,AccuracyGME,'LineWidth',2)
plot(epsilonList,AccuracyCWA,'LineWidth',2)
legend('REC_{TPR}','REC_{TNR}','REC_{GME}','REC_{CWA}')
xlabel('Error Tolerance \epsilon')
ylabel('Validation Accurancy')
title('Synthetic Dataset: r = 1.7')
subtitle('Training Cost Function = CWE')
set(gcf,'color','white')


% Plot Just W with GMSERA
fig = figure;
plot(epsilonList,AccuracyTPR,'LineWidth',2)
hold on
plot(epsilonListSERA,AccuracyTPRSERA,'LineWidth',2)
legend('CWE_{TPR}','SERA_{TPR}')
xlabel('Error Tolerance \epsilon')
ylabel('Validation Accurancy')
title('Synthetic Dataset: r = 1.7')
set(gcf,'color','white')




%% CWE Sweeps 2
model = 'LinearRegression';
costFunctions = [
    "MAE"
    "RR"
    "PLOSS"
    "SERA"
    "BMSE"
    "KRR"
    "GME"
    "CWE"
    ];

hyp.w = .5; %Used by CWE
hyp.thresh = 90; %Percentile; Used by CWE,GME
hyp.sigma = .1; %Used by BMSE
hyp.lambda = 5e-3; %Used by RR and KRR
hyp.kernel = 'RBF'; % Options: linear or RBF
hyp.k_sigma = .1; %Must be > 0. Used by KRR-RBF
hyp.c = 0; % Must be >= 0. Used by KRR-linear

wList = 0:.1:1;

for c = 1:length(costFunctions)
    costFunction = costFunctions(c);
    [YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = loadPreds(dataSet,r_value,model,costFunction);
    costGME(c) = calculateCost('GME',YPred_val,YVal,hyp);
    for w = 1:length(wList)
        hyp.w = wList(w);
        costCWE(c,w) = calculateCost('CWE',YPred_val,YVal,hyp);
    end
end


% Plot
figure
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    if i > 6
        plot(wList,costCWE(i,:),'Color',color,'DisplayName',costFunctions{i},'LineWidth',2) 
    else
        plot(wList,costCWE(i,:),'Color',color,'DisplayName',costFunctions{i},'LineWidth',1)
    end
    hold on
end
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    scatter(0.5,costGME(i),36,color,'Filled') 
    hold on
end
set(gcf,'color','white')
xlabel('w')
ylabel('Validation Error')
title('Synthetic Dataset: r = 1.7')



% Create Legend
figure
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    if i > 6
        plot(wList,costCWE(i,:),'Color',color,'DisplayName',costFunctions{i},'LineWidth',2) 
    else
        plot(wList,costCWE(i,:),'Color',color,'DisplayName',costFunctions{i},'LineWidth',1)
    end
    hold on
end
legend('Location','Best')
set(gcf,'color','white')

figure
for i = 1:length(costFunctions)
    color = colorsOrdered(i);
    scatter(0.5,costGME(i),36,color,'Filled','DisplayName',costFunctions{i}) 
    hold on
end
legend