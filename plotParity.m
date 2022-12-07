% Plot parity plot

function plotParity(YTrain,YPred, path)

    fig = figure('Visible','off');
    scatter(YTrain,YPred,'filled','MarkerFaceAlpha',.3,'MarkerEdgeAlpha',.3)
    hold on
    rangeVal = range([YTrain,YPred]);
    minVal = min([YTrain,YPred]);
    minVal = minVal - rangeVal*.1;
    maxVal = max([YTrain,YPred]);
    maxVal = maxVal + rangeVal*.1;
    ylim([minVal,maxVal])
    xlim([minVal,maxVal])
    plot([minVal,maxVal],[minVal,maxVal],'r--')
    set(gcf,'color','white')
    legend('Data Points','y = x Reference Line','Location','NorthWest')
    xlabel('Actual')
    ylabel('Predicted')
    savefig(path);
    saveas(fig,strcat(path,'.png'))

end