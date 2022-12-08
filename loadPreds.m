% Loads Predictions that have previusly been calculated


function [YPred_train,YTrain,YPred_val,YVal,YPred_test,YTest] = loadPreds(dataSet,r_value,model,costFunction)

    dataStr = dataSet;
    if strcmp(dataSet, "synthetic")
        dataStr = strcat(dataSet,'_r=',num2str(r_value));
    end
    path = strcat('results/', model,'_', dataStr, '_', costFunction);
    path_preds = strcat(path,'/preds.mat');
    load(path_preds, 'YPred_train','YTrain','YPred_val','YVal','YPred_test','YTest')

end