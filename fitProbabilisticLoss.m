function phi_y = fitProbabilisticLoss(YTrain ,print_graph)
    [f,xi] = ksdensity(YTrain);
    if print_graph
        histogram(YTrain);
        yyaxis left
        hold on;
        yyaxis right
        plot(xi,f,'LineWidth',2);
        [fnorm,fmin,frange] = norm_zero2one(f);
        f_reversed = 1-fnorm;
        plot(xi,f_reversed,'LineWidth',2);
        title('KDE of Normalized Prices');
        xlabel('Normalized Price');
        legend('data frequency', 'KDE', 'Ploss')
    end
    phi_y = interp1(xi, f, YTrain);
end


function [norm_data,minData,rangeData] = norm_zero2one(data)
    minData = min(data);
    rangeData = range(data);
    norm_data  = (data - minData)./rangeData;
end