function [XTrain, YTrain, XVal, YVal, XTest, YTest] = preprocess_save_data(name)
    switch name
        case 'cali'
            data = get_cali_data();
    end
    split_ratio = [0.7,0.15,0.15];
    [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_data(data, split_ratio);
    file_name = convertStringsToChars(strcat('data/',strcat(name + ".mat")));
    save(file_name, 'XTrain', 'YTrain', 'XVal', 'YVal', 'XTest', 'YTest');
    disp('saved preprocessed data under data folder');
end

function [XTrain, YTrain, XVal, YVal, XTest, YTest] = split_data(data, split_ratio)
    n = size(data,1);
    n_train = floor(n * split_ratio(1));
    
    start_index_val = n_train + 1;
    
    n_val = floor(n * split_ratio(2));
    
    start_index_test = n_train + n_val +1;
    
    X_table = data(:,1:size(data,2)-1);
    Y_table = data(:,end);

    X = X_table{:,:};
    Y = Y_table{:,:};

    XTrain = X(1:n_train,:); 
    YTrain = Y(1:n_train,:);
    XVal = X(start_index_val:start_index_test-1,:);
    YVal = Y(start_index_val:start_index_test-1,:);
    XTest = X(start_index_test:end,:);
    YTest = Y(start_index_test:end,:);
    
end

function cali_data = get_cali_data()
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
    
    % Normalize data
    cali_data = normalize(cali_data);

    % Shuffle the data
    shuffled_array = cali_data(randperm(size(cali_data,1)),:);
    cali_data = shuffled_array;
end