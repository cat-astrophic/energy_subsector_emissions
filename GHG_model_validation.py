# This script tests to verify that tbe methodology used in the paper is valid

# Importing required modules

import pandas as pd
import statsmodels.api as stats

# Declaring filepaths for data files

arpath = 'C:/Users/User/Documents/Data/state_level_ardata.csv'

# Structuring the dataframes for regression

data = pd.read_csv(arpath)
Y = data['Energy']
X_EKC = data[['Energy_Lag', 'GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Linear = data[['Energy_Lag', 'GDP_per_capita', 'Population_Density', 'Renewables', 'HDD', 'CDD']]

# Running AR-1 regression modes with EKC Hypothesis format

ekc_model = stats.OLS(Y.astype(float), X_EKC.astype(float))
ekc_results = ekc_model.fit()
print(ekc_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/ekc_method_testing.txt', 'w')
file.write(ekc_results.summary().as_text())
file.close()

# Running AR-1 regression modes with Linear Growth Hypothesis format

linear_model = stats.OLS(Y.astype(float), X_Linear.astype(float))
linear_results = linear_model.fit()
print(linear_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/linear_method_testing.txt', 'w')
file.write(linear_results.summary().as_text())
file.close()

# Output results to the researcher

if ekc_results.conf_int()[0][0] < 1 and ekc_results.conf_int()[1][0] > 1 and linear_results.conf_int()[0][0] < 1 and linear_results.conf_int()[1][0] > 1:
    
    print('\n\nModel approach validated!')

else:
    print('\n\nModel approach NOT VALID!')

