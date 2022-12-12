Full code is available at: https://github.com/acunfatih/503_project
Github Repository should be public and available to anyone. However, if you have trouble getting in, contact Fatih. 
California dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices


Designed to run on Matlab 2021a or above. 


1. synthetic_data_2d.m: Used to generate the synthetic datasets with various levels of skew.
2. preprocess_save_data.m: This function contains the function get_cali_data which imports and cleans the california dataset. It is not necssary to run this, as it is run automatically when importing the data using scripts 3-5 below.
3. HyparameterOptimizer_#.m: Five scripts that help the user select the hyperparameters before generating the main data. The selected hyperparameter values should be put into MainScriptIterator.m.
4. MainScriptIterator.m is the main script iterating over all the models, costfunctions and the datasets. The results(plots, predictions) are saved to the related directories under results folder. results/results.csv file is to store the evaluation results for each run. If you want to do a single run, pick one model, costFunction and dataset by commenting out the others. Warning: Running all datasets can take ~1.5 hours.
5. SlideExplanations.m: Creates many plots and figures that were used in the final report and presentation.




