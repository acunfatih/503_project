function [XTrain, YTrain, XVal, YVal, XTest, YTest,minData,rangeData] = preprocess_save_data(name)
    switch name
        case 'cali'
            [data, max_label,minData,rangeData] = get_cali_data();
        case 'synthetic_r1.1'
            [data, max_label,minData,rangeData] = get_synt_data(1.1)
        case 'synthetic_r1.2'
            [data, max_label,minData,rangeData] = get_synt_data(1.2)
        case 'synthetic_r1.3'
            [data, max_label,minData,rangeData] = get_synt_data(1.3)
        case 'synthetic_r1.4'
            [data, max_label,minData,rangeData] = get_synt_data(1.4)
        case 'synthetic_r1.5'
            [data, max_label,minData,rangeData] = get_synt_data(1.5)
        case 'synthetic_r1.6'
            [data, max_label,minData,rangeData] = get_synt_data(1.6)
        case 'synthetic_r1.7'
            [data, max_label,minData,rangeData] = get_synt_data(1.7)
        case 'synthetic_r1.8'
            [data, max_label,minData,rangeData] = get_synt_data(1.8)
        case 'synthetic_r1.9'
            [data, max_label,minData,rangeData] = get_synt_data(1.9)
        case 'synthetic_r2.0'
            [data, max_label,minData,rangeData] = get_synt_data(2.0)
            
    end
    split_ratio = [0.7,0.15,0.15];
    [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_data(data, split_ratio);
    file_name = strcat('data/',strcat(name + ".mat"));
    save(convertStringsToChars(file_name), 'XTrain', 'YTrain', 'XVal', 'YVal', 'XTest', 'YTest','minData','rangeData');    

    save(convertStringsToChars(strcat('data/max_label_', strcat(name + ".mat"))), 'max_label');
    
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

function [cali_data,max_label,minData,rangeData] = get_cali_data()
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
    max_label = max(cali_data(:,end));
    cali_data(:,1) = abs(cali_data(:,1));
    [cali_data,minData,rangeData] = norm_zero2one(cali_data);
    cali_data = norm_zero2one(cali_data);

    % Shuffle the data
    shuffled_array = cali_data(randperm(size(cali_data,1)),:);
    cali_data = shuffled_array;
end

function [norm_data,minData,rangeData] = norm_zero2one(data)
    minData = min(data);
    rangeData = range(data);
    norm_data  = (data - minData)./rangeData;
end

function [synt_data,max_label,minData,rangeData] = get_synt_data(r_value)

    file_name = sprintf("data/2D_sdata1_r%.2f.csv",r_value);
    synt_data_raw = readtable(file_name);
    
    synt_data =  table2array(synt_data_raw);
    synt_label = synt_data(:,3);
    
    %Normalization?
    max_label = max(synt_label);
    minData = min(synt_data);
    rangeData = range(synt_data);
    
end
