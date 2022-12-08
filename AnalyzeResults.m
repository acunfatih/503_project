% Analyze Results
clear all
close all
clc

% load data
T = readtable('results/resultsNoTime.csv');
%% MSE
r = 1.7;
matchVal = sprintf('synthetic_r=%.1f',r);
idxr = strcmp(T.dataStr,matchVal);

figure
bar(T.MSE_val(idxr))
title(sprintf('MSE Validation Cost: %s',matchVal))
xlabel('Training Cost Function')
ylabel('Error')
set(gca,'xticklabel',T.costFunction(idxr))
set(gcf,'color','white')


%% BMSE
r = 1.7;
matchVal = sprintf('synthetic_r=%.1f',r);
idxr = strcmp(T.dataStr,matchVal);

figure
bar(T.BMSE_val(idxr))
title(sprintf('BMSE Validation Cost: %s',matchVal))
xlabel('Training Cost Function')
ylabel('Error')
set(gca,'xticklabel',T.costFunction(idxr))
set(gcf,'color','white')



%% CWE
r = 1.7;
matchVal = sprintf('synthetic_r=%.1f',r);
idxr = strcmp(T.dataStr,matchVal);

figure
bar(T.CWE_val(idxr))
title(sprintf('CWE Validation Cost: %s',matchVal))
xlabel('Training Cost Function')
ylabel('Error')
set(gca,'xticklabel',T.costFunction(idxr))
set(gcf,'color','white')