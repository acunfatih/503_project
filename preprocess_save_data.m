% loads the data from csv, preprocesses and saves as .mat under 'data' folder

function [XTrain, YTrain, XVal, YVal, XTest, YTest,minData,rangeData] = preprocess_save_data(name,r)
    switch name
        case 'cali'
            [data, maxLabel,minData,rangeData] = get_cali_data();
            split_ratio = [0.7,0.15,0.15];
            [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_data(data, split_ratio);
        case 'synthetic'
            [data, maxLabel,minData,rangeData] = get_synt_data(r);
            [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_synt(data);
            
    end
    name2 = strcat(name,num2str(r,'%.1f'));
    file_name = strcat('data/',strcat(name2 + ".mat"));
    save(convertStringsToChars(file_name), 'XTrain', 'YTrain', 'XVal', 'YVal', 'XTest', 'YTest','minData','rangeData', 'maxLabel');    

%     save(convertStringsToChars(strcat('data/max_label_', strcat(name + ".mat"))), 'max_label');
    
    disp('saved preprocessed data under data folder');

end

function [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_data(data, split_ratio)
    n = size(data,1);
    n_train = floor(n * split_ratio(1));
    
    start_index_val = n_train + 1;
    
    n_val = floor(n * split_ratio(2));
    
    start_index_test = n_train + n_val +1;

    X = data(:,1:size(data,2)-1);
    Y = data(:,end);

    XTrain = X(1:n_train,:)'; 
    YTrain = Y(1:n_train,:)';
    XVal = X(start_index_val:start_index_test-1,:)';
    YVal = Y(start_index_val:start_index_test-1,:)';
    XTest = X(start_index_test:end,:)';
    YTest = Y(start_index_test:end,:)';
    
end

%
function [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_synt(data)
    XTrain = data([1:700],[1:2])';
    YTrain = data([1:700],3)';
    
    XVal = data([701:850],[1:2])';
    YVal = data([701:850],3)';
    
    XTest = data([851:1000],[1:2])';
    YTest = data([851:1000],3)';
    
end

function [cali_data,maxLabel,minData,rangeData] = get_cali_data()
    cali_data_raw = readtable("data/california_housing.csv");
    % 'dropping ocean_proximity column'
    cali_data_raw.ocean_proximity = [];
    
    % drop rows with nan
    cali_data = rmmissing(cali_data_raw);
    
    total_rooms_pop_norm = cali_data.total_rooms./cali_data.population;
    total_bedrooms_pop_norm = cali_data.total_bedrooms./cali_data.population;
    cali_data.total_rooms_pop_norm = total_rooms_pop_norm;
    cali_data.total_bedrooms_pop_norm = total_bedrooms_pop_norm;
    cali_data = [cali_data(:,1:5) cali_data(:,10:11) cali_data(:,6:9)];
    cali_data = cali_data{:,:};
    
    % Normalize data
    maxLabel = max(cali_data(:,end));
    cali_data(:,1) = abs(cali_data(:,1));
    [cali_data,minData,rangeData] = norm_zero2one(cali_data);
    cali_data = norm_zero2one(cali_data);

    % Shuffle the data
    shuffled_array = cali_data(randperm(size(cali_data,1)),:);
    cali_data = shuffled_array;
end



function [synt_data,maxLabel,minData,rangeData] = get_synt_data(r_value)

    file_name = sprintf("data/2D_sdata1_r%.2f.csv",r_value);
    synt_data_raw = readtable(file_name);
    
    synt_data =  table2array(synt_data_raw);
    synt_label = synt_data(:,3); %extracting labels
    
    %Normalization
    maxLabel = max(synt_label);
    [synt_data,minData,rangeData] = norm_zero2one(synt_data);
end


function [norm_data,minData,rangeData] = norm_zero2one(data)
    minData = min(data);
    rangeData = range(data);
    norm_data  = (data - minData)./rangeData;
end

