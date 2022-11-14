function [XTrain, YTrain, Xval, Yval] = load_dataset(name)
    switch name
        case 'cali'
            data = get_cali_data();
    end

    [XTrain, YTrain, Xval, Yval] = split_data(data, 0.8);
    
end


function [XTrain, YTrain, Xval, Yval] = split_data(data, split_ratio)
    
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