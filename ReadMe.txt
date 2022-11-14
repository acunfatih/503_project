--How to run a model and cost function
1. Open MainScript.m
2. Pick dataset to work with
3. Select model and cost function
4. Press Run

Note: there are two different minimizing functions that can be used: fminsearch and fminunc. 
The current system uses both and then selects the theta with the lowest cost. There are also options that can be tune for these minimizing functions.


--How to add a model
1. Add theta0 initialization value into initializeTheta.m
2. Add formula for predicting Y from Theta in predictY.m



--How to add a cost function
1. Add cost function to calculateCost.m