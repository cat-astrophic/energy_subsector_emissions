# Running various regression models for state level analyses

# Importing required modules

import numpy as np
import pandas as pd
import statsmodels.api as stats
import statsmodels.formula.api as sform
import matplotlib.pyplot as plt
from scipy.stats import norm

# Declaring filepaths for data file

filepath = 'C:/Users/User/Documents/Data/state_level_data.csv'

# Structuring the dataframes for regression

# Read file with pandas and create dummies for the WRI data set

data = pd.read_csv(filepath)
year_dummies = pd.get_dummies(data['Year'])
state_dummies = pd.get_dummies(data['State'])

# Creating the base model variables dataframe

X = data[['GDP_per_capita', 'GDP_per_capita_2', 'GDP_per_capita_3', 'Population_Density']]
X = stats.add_constant(X)

X2 = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density']]
X2 = stats.add_constant(X2)

# Creating the fixed effects model variables dataframes

year_X = pd.concat([X, year_dummies], axis = 1)
year_X.drop([1990], inplace = True, axis = 1)
fixed_X = pd.concat([year_X, state_dummies], axis = 1)
fixed_X.drop(['California'], inplace = True, axis = 1)

year_X2 = pd.concat([X2, year_dummies], axis = 1)
year_X2.drop([1990], inplace = True, axis = 1)
fixed_X2 = pd.concat([year_X2, state_dummies], axis = 1)
fixed_X2.drop(['California'], inplace = True, axis = 1)

# Creating the dependent variable dataframes

Y_Energy = data['Energy']
Y_Commercial = data['Commercial']
Y_Residential = data['Residential']
Y_Industrial = data['Industrial']
Y_Transportation = data['Transportation']
Y_Electric_Power = data['Electric_Power']
Y_Fugitive = data['Fugitive']

# Creating the random effects model dataframe

rand_X = pd.concat([year_X, data['State']], axis = 1)

rand_X2 = pd.concat([year_X2, data['State']], axis = 1)

# All regression models with cubic term included

# Basic OLS models

# Emissions from energy

energy_model = stats.OLS(Y_Energy.astype(float), X.astype(float))
energy_results = energy_model.fit()
print(energy_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_model.txt', 'w')
file.write(energy_results.summary().as_text())
file.close()

# Emissions from commercial

commercial_model = stats.OLS(Y_Commercial.astype(float), X.astype(float))
commercial_results = commercial_model.fit()
print(commercial_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_model.txt', 'w')
file.write(commercial_results.summary().as_text())
file.close()

# Emissions from residential

residential_model = stats.OLS(Y_Residential.astype(float), X.astype(float))
residential_results = residential_model.fit()
print(residential_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_model.txt', 'w')
file.write(residential_results.summary().as_text())
file.close()

# Emissions from industry

industrial_model = stats.OLS(Y_Industrial.astype(float), X.astype(float))
industrial_results = industrial_model.fit()
print(industrial_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industrial_model.txt', 'w')
file.write(industrial_results.summary().as_text())
file.close()

# Emissions from transportation

transportation_model = stats.OLS(Y_Transportation.astype(float), X.astype(float))
transportation_results = transportation_model.fit()
print(transportation_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_model.txt', 'w')
file.write(transportation_results.summary().as_text())
file.close()

# Emissions from electric power

electric_model = stats.OLS(Y_Electric_Power.astype(float), X.astype(float))
electric_results = electric_model.fit()
print(electric_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_model.txt', 'w')
file.write(electric_results.summary().as_text())
file.close()

# Fugitive emissions in energy

fugitive_model = stats.OLS(Y_Fugitive.astype(float), X.astype(float))
fugitive_results = fugitive_model.fit()
print(fugitive_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_model.txt', 'w')
file.write(fugitive_results.summary().as_text())
file.close()

# Regression models with fixed effects for year only

# Emissions from energy

energy_year_model = stats.OLS(Y_Energy.astype(float), year_X.astype(float))
energy_year_results = energy_year_model.fit()
print(energy_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_year_model.txt', 'w')
file.write(energy_year_results.summary().as_text())
file.close()

# Emissions from commercial

commercial_year_model = stats.OLS(Y_Commercial.astype(float), year_X.astype(float))
commercial_year_results = commercial_year_model.fit()
print(commercial_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_year_model.txt', 'w')
file.write(commercial_year_results.summary().as_text())
file.close()

# Emissions from residential

residential_year_model = stats.OLS(Y_Residential.astype(float), year_X.astype(float))
residential_year_results = residential_year_model.fit()
print(residential_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_year_model.txt', 'w')
file.write(residential_year_results.summary().as_text())
file.close()

# Emissions from industry

industrial_year_model = stats.OLS(Y_Industrial.astype(float), year_X.astype(float))
industrial_year_results = industrial_year_model.fit()
print(industrial_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industrial_year_model.txt', 'w')
file.write(industrial_year_results.summary().as_text())
file.close()

# Emissions from transportation

transportation_year_model = stats.OLS(Y_Transportation.astype(float), year_X.astype(float))
transportation_year_results = transportation_year_model.fit()
print(transportation_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_year_model.txt', 'w')
file.write(transportation_year_results.summary().as_text())
file.close()

# Emissions from electric power

electric_year_model = stats.OLS(Y_Electric_Power.astype(float), year_X.astype(float))
electric_year_results = electric_year_model.fit()
print(electric_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_year_model.txt', 'w')
file.write(electric_year_results.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_year_model = stats.OLS(Y_Fugitive.astype(float), year_X.astype(float))
fugitive_year_results = fugitive_year_model.fit()
print(fugitive_year_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_year_model.txt', 'w')
file.write(fugitive_year_results.summary().as_text())
file.close()

# Regression models with fixed effects for year and state

# Emissions from energy

energy_fixed_model = stats.OLS(Y_Energy.astype(float), fixed_X.astype(float))
energy_fixed_results = energy_fixed_model.fit()
print(energy_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_fixed_model.txt', 'w')
file.write(energy_fixed_results.summary().as_text())
file.close()

# Emissions from commercial

commercial_fixed_model = stats.OLS(Y_Commercial.astype(float), fixed_X.astype(float))
commercial_fixed_results = commercial_fixed_model.fit()
print(commercial_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_fixed_model.txt', 'w')
file.write(commercial_fixed_results.summary().as_text())
file.close()

# Emissions from residential

residential_fixed_model = stats.OLS(Y_Residential.astype(float), fixed_X.astype(float))
residential_fixed_results = residential_fixed_model.fit()
print(residential_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_fixed_model.txt', 'w')
file.write(residential_fixed_results.summary().as_text())
file.close()

# Emissions from industry

industrial_fixed_model = stats.OLS(Y_Industrial.astype(float), fixed_X.astype(float))
industrial_fixed_results = industrial_fixed_model.fit()
print(industrial_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industrial_fixed_model.txt', 'w')
file.write(industrial_fixed_results.summary().as_text())
file.close()

# Emissions from transportation

transportation_fixed_model = stats.OLS(Y_Transportation.astype(float), fixed_X.astype(float))
transportation_fixed_results = transportation_fixed_model.fit()
print(transportation_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_fixed_model.txt', 'w')
file.write(transportation_fixed_results.summary().as_text())
file.close()

# Emissions from electric power

electric_fixed_model = stats.OLS(Y_Electric_Power.astype(float), fixed_X.astype(float))
electric_fixed_results = electric_fixed_model.fit()
print(electric_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_fixed_model.txt', 'w')
file.write(electric_fixed_results.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_fixed_model = stats.OLS(Y_Fugitive.astype(float), fixed_X.astype(float))
fugitive_fixed_results = fugitive_fixed_model.fit()
print(fugitive_fixed_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_fixed_model.txt', 'w')
file.write(fugitive_fixed_results.summary().as_text())
file.close()

# All regression models without cubic term included

# Basic OLS models

# Emissions from energy

energy_model2 = stats.OLS(Y_Energy.astype(float), X2.astype(float))
energy_results2 = energy_model2.fit()
print(energy_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_model2.txt', 'w')
file.write(energy_results2.summary().as_text())
file.close()

# Emissions from commercial

commercial_model2 = stats.OLS(Y_Commercial.astype(float), X2.astype(float))
commercial_results2 = commercial_model2.fit()
print(commercial_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_model2.txt', 'w')
file.write(commercial_results2.summary().as_text())
file.close()

# Emissions from residential

residential_model2 = stats.OLS(Y_Residential.astype(float), X2.astype(float))
residential_results2 = residential_model2.fit()
print(residential_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_model2.txt', 'w')
file.write(residential_results2.summary().as_text())
file.close()

# Emissions from industry

industrial_model2 = stats.OLS(Y_Industrial.astype(float), X2.astype(float))
industrial_results2 = industrial_model2.fit()
print(industrial_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industrial_model2.txt', 'w')
file.write(industrial_results2.summary().as_text())
file.close()

# Emissions from transportation

transportation_model2 = stats.OLS(Y_Transportation.astype(float), X2.astype(float))
transportation_results2 = transportation_model2.fit()
print(transportation_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_model2.txt', 'w')
file.write(transportation_results2.summary().as_text())
file.close()

# Emissions from electric power

electric_model2 = stats.OLS(Y_Electric_Power.astype(float), X2.astype(float))
electric_results2 = electric_model2.fit()
print(electric_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_model2.txt', 'w')
file.write(electric_results2.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_model2 = stats.OLS(Y_Fugitive.astype(float), X2.astype(float))
fugitive_results2 = fugitive_model2.fit()
print(fugitive_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_model2.txt', 'w')
file.write(fugitive_results2.summary().as_text())
file.close()

# Regression models with fixed effects for year only

# Emissions from agriculture

# Emissions from energy

energy_year_model2 = stats.OLS(Y_Energy.astype(float), year_X2.astype(float))
energy_year_results2 = energy_year_model2.fit()
print(energy_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_year_model2.txt', 'w')
file.write(energy_year_results2.summary().as_text())
file.close()

# Emissions from commercial

commercial_year_model2 = stats.OLS(Y_Commercial.astype(float), year_X2.astype(float))
commercial_year_results2 = commercial_year_model2.fit()
print(commercial_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_year_model2.txt', 'w')
file.write(commercial_year_results2.summary().as_text())
file.close()

# Emissions from residential

residential_year_model2 = stats.OLS(Y_Residential.astype(float), year_X2.astype(float))
residential_year_results2 = residential_year_model2.fit()
print(residential_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_year_model2.txt', 'w')
file.write(residential_year_results2.summary().as_text())
file.close()

# Emissions from industry

industrial_year_model2 = stats.OLS(Y_Industrial.astype(float), year_X2.astype(float))
industrial_year_results2 = industrial_year_model2.fit()
print(industrial_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industrial_year_model2.txt', 'w')
file.write(industrial_year_results2.summary().as_text())
file.close()

# Emissions from transportation

transportation_year_model2 = stats.OLS(Y_Transportation.astype(float), year_X2.astype(float))
transportation_year_results2 = transportation_year_model2.fit()
print(transportation_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_year_model2.txt', 'w')
file.write(transportation_year_results2.summary().as_text())
file.close()

# Emissions from electric power

electric_year_model2 = stats.OLS(Y_Electric_Power.astype(float), year_X2.astype(float))
electric_year_results2 = electric_year_model2.fit()
print(electric_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_year_model2.txt', 'w')
file.write(electric_year_results2.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_year_model2 = stats.OLS(Y_Fugitive.astype(float), year_X2.astype(float))
fugitive_year_results2 = fugitive_year_model2.fit()
print(fugitive_year_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_year_model2.txt', 'w')
file.write(fugitive_year_results2.summary().as_text())
file.close()

# Regression models with fixed effects for year and state

# Emissions from energy

energy_fixed_model2 = stats.OLS(Y_Energy.astype(float), fixed_X2.astype(float))
energy_fixed_results2 = energy_fixed_model2.fit()
print(energy_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_fixed_model2.txt', 'w')
file.write(energy_fixed_results2.summary().as_text())
file.close()

# Emissions from commercial

commercial_fixed_model2 = stats.OLS(Y_Commercial.astype(float), fixed_X2.astype(float))
commercial_fixed_results2 = commercial_fixed_model2.fit()
print(commercial_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_fixed_model2.txt', 'w')
file.write(commercial_fixed_results2.summary().as_text())
file.close()

# Emissions from residential

residential_fixed_model2 = stats.OLS(Y_Residential.astype(float), fixed_X2.astype(float))
residential_fixed_results2 = residential_fixed_model2.fit()
print(residential_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_fixed_model2.txt', 'w')
file.write(residential_fixed_results2.summary().as_text())
file.close()

# Emissions from industry

industrial_fixed_model2 = stats.OLS(Y_Industrial.astype(float), fixed_X2.astype(float))
industrial_fixed_results2 = industrial_fixed_model2.fit()
print(industrial_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industrial_fixed_model2.txt', 'w')
file.write(industrial_fixed_results2.summary().as_text())
file.close()

# Emissions from transportation

transportation_fixed_model2 = stats.OLS(Y_Transportation.astype(float), fixed_X2.astype(float))
transportation_fixed_results2 = transportation_fixed_model2.fit()
print(transportation_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_fixed_model2.txt', 'w')
file.write(transportation_fixed_results2.summary().as_text())
file.close()

# Emissions from electric power

electric_fixed_model2 = stats.OLS(Y_Electric_Power.astype(float), fixed_X2.astype(float))
electric_fixed_results2 = electric_fixed_model2.fit()
print(electric_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_fixed_model2.txt', 'w')
file.write(electric_fixed_results2.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_fixed_model2 = stats.OLS(Y_Fugitive.astype(float), fixed_X2.astype(float))
fugitive_fixed_results2 = fugitive_fixed_model2.fit()
print(fugitive_fixed_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_fixed_model2.txt', 'w')
file.write(fugitive_fixed_results2.summary().as_text())
file.close()

# Random effects regression models

# Emissions from agriculture

# Emissions from energy

energy_rand = pd.concat([Y_Energy, rand_X], axis = 1)
energy_rand.drop(['const'], inplace = True, axis = 1)
energy_rand_specification = 'Energy ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
energy_rand_model = sform.mixedlm(energy_rand_specification, data = energy_rand, groups = 'State')
energy_rand_results = energy_rand_model.fit()
print(energy_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_rand_model.txt', 'w')
file.write(energy_rand_results.summary().as_text())
file.close()

# Emissions from commercial

commercial_rand = pd.concat([Y_Commercial, rand_X], axis = 1)
commercial_rand.drop(['const'], inplace = True, axis = 1)
commercial_rand_specification = 'Commercial ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
commercial_rand_model = sform.mixedlm(commercial_rand_specification, data = commercial_rand, groups = 'State')
commercial_rand_results = commercial_rand_model.fit()
print(commercial_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_rand_model.txt', 'w')
file.write(commercial_rand_results.summary().as_text())
file.close()

# Emissions from residential

residential_rand = pd.concat([Y_Residential, rand_X], axis = 1)
residential_rand.drop(['const'], inplace = True, axis = 1)
residential_rand_specification = 'Residential ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
residential_rand_model = sform.mixedlm(residential_rand_specification, data = residential_rand, groups = 'State')
residential_rand_results = residential_rand_model.fit()
print(residential_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_rand_model.txt', 'w')
file.write(residential_rand_results.summary().as_text())
file.close()

# Emissions from industry

industry_rand = pd.concat([Y_Industrial, rand_X], axis = 1)
industry_rand.drop(['const'], inplace = True, axis = 1)
industry_rand_specification = 'Industrial ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
industry_rand_model = sform.mixedlm(industry_rand_specification, data = industry_rand, groups = 'State')
industry_rand_results = industry_rand_model.fit()
print(industry_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industry_rand_model.txt', 'w')
file.write(industry_rand_results.summary().as_text())
file.close()

# Emissions from transportation

transportation_rand = pd.concat([Y_Transportation, rand_X], axis = 1)
transportation_rand.drop(['const'], inplace = True, axis = 1)
transportation_rand_specification = 'Transportation ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
transportation_rand_model = sform.mixedlm(transportation_rand_specification, data = transportation_rand, groups = 'State')
transportation_rand_results = transportation_rand_model.fit()
print(transportation_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_rand_model.txt', 'w')
file.write(transportation_rand_results.summary().as_text())
file.close()

# Emissions from electric power

electric_rand = pd.concat([Y_Electric_Power, rand_X], axis = 1)
electric_rand.drop(['const'], inplace = True, axis = 1)
electric_rand_specification = 'Electric_Power ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
electric_rand_model = sform.mixedlm(electric_rand_specification, data = electric_rand, groups = 'State')
electric_rand_results = electric_rand_model.fit()
print(electric_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_rand_model.txt', 'w')
file.write(electric_rand_results.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_rand = pd.concat([Y_Fugitive, rand_X], axis = 1)
fugitive_rand.drop(['const'], inplace = True, axis = 1)
fugitive_rand_specification = 'Fugitive ~ GDP_per_capita + GDP_per_capita_2 + GDP_per_capita_3 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
fugitive_rand_model = sform.mixedlm(fugitive_rand_specification, data = fugitive_rand, groups = 'State')
fugitive_rand_results = fugitive_rand_model.fit()
print(fugitive_rand_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_rand_model.txt', 'w')
file.write(fugitive_rand_results.summary().as_text())
file.close()

# Random effects regression models without cubic term

# Emissions from energy

energy_rand2 = pd.concat([Y_Energy, rand_X2], axis = 1)
energy_rand2.drop(['const'], inplace = True, axis = 1)
energy_rand_specification2 = 'Energy ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
energy_rand_model2 = sform.mixedlm(energy_rand_specification2, data = energy_rand2, groups = 'State')
energy_rand_results2 = energy_rand_model2.fit()
print(energy_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/energy_rand_model2.txt', 'w')
file.write(energy_rand_results2.summary().as_text())
file.close()

# Emissions from commercial

commercial_rand2 = pd.concat([Y_Commercial, rand_X2], axis = 1)
commercial_rand2.drop(['const'], inplace = True, axis = 1)
commercial_rand_specification2 = 'Commercial ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
commercial_rand_model2 = sform.mixedlm(commercial_rand_specification2, data = commercial_rand2, groups = 'State')
commercial_rand_results2 = commercial_rand_model2.fit()
print(commercial_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/commercial_rand_model2.txt', 'w')
file.write(commercial_rand_results2.summary().as_text())
file.close()

# Emissions from residential

residential_rand2 = pd.concat([Y_Residential, rand_X2], axis = 1)
residential_rand2.drop(['const'], inplace = True, axis = 1)
residential_rand_specification2 = 'Residential ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
residential_rand_model2 = sform.mixedlm(residential_rand_specification2, data = residential_rand2, groups = 'State')
residential_rand_results2 = residential_rand_model2.fit()
print(residential_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/residential_rand_model2.txt', 'w')
file.write(residential_rand_results2.summary().as_text())
file.close()

# Emissions from industry

industry_rand2 = pd.concat([Y_Industrial, rand_X2], axis = 1)
industry_rand2.drop(['const'], inplace = True, axis = 1)
industry_rand_specification2 = 'Industrial ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
industry_rand_model2 = sform.mixedlm(industry_rand_specification2, data = industry_rand2, groups = 'State')
industry_rand_results2 = industry_rand_model2.fit()
print(industry_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/industry_rand_model2.txt', 'w')
file.write(industry_rand_results2.summary().as_text())
file.close()

# Emissions from transportation

transportation_rand2 = pd.concat([Y_Transportation, rand_X2], axis = 1)
transportation_rand2.drop(['const'], inplace = True, axis = 1)
transportation_rand_specification2 = 'Transportation ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
transportation_rand_model2 = sform.mixedlm(transportation_rand_specification2, data = transportation_rand2, groups = 'State')
transportation_rand_results2 = transportation_rand_model2.fit()
print(transportation_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/transportation_rand_model2.txt', 'w')
file.write(transportation_rand_results2.summary().as_text())
file.close()

# Emissions from electric power

electric_rand2 = pd.concat([Y_Electric_Power, rand_X2], axis = 1)
electric_rand2.drop(['const'], inplace = True, axis = 1)
electric_rand_specification2 = 'Electric_Power ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
electric_rand_model2 = sform.mixedlm(electric_rand_specification2, data = electric_rand2, groups = 'State')
electric_rand_results2 = electric_rand_model2.fit()
print(electric_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/electric_rand_model2.txt', 'w')
file.write(electric_rand_results2.summary().as_text())
file.close()

# Fugitive emissions from energy

fugitive_rand2 = pd.concat([Y_Fugitive, rand_X2], axis = 1)
fugitive_rand2.drop(['const'], inplace = True, axis = 1)
fugitive_rand_specification2 = 'Fugitive ~ GDP_per_capita + GDP_per_capita_2 + Population_Density + Q(1991) + Q(1992) + Q(1993) + Q(1994) + Q(1995) + Q(1996) + Q(1997) + Q(1998) + Q(1999) + Q(2000) + Q(2001) + Q(2002) + Q(2003) + Q(2004) + Q(2005) + Q(2006) + Q(2007) + Q(2008) + Q(2009) + Q(2010) + Q(2011)'
fugitive_rand_model2 = sform.mixedlm(fugitive_rand_specification2, data = fugitive_rand2, groups = 'State')
fugitive_rand_results2 = fugitive_rand_model2.fit()
print(fugitive_rand_results2.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_rand_model2.txt', 'w')
file.write(fugitive_rand_results2.summary().as_text())
file.close()

# Forecasting Population Density

# Generate list of states and obtain parameters for forecasts

States = []

for state in data.State:
    if state not in States:
        States.append(state)

pop_params = [[],[]]

y = [i for i in range(22)]
y = stats.add_constant(y)

for state in States:
    pop_temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            pop_temp.append(data.Population_Density[i])
    temp_pop_mod = stats.OLS(pop_temp, y)
    temp_pop_res = temp_pop_mod.fit()
    pop_params[0].append(temp_pop_res.params[0])
    pop_params[1].append(temp_pop_res.params[1])

# Create dataframe with pandas to record trends

d = {'State':States, 'PD_int':pop_params[0], 'PD_slope':pop_params[1]}
params_df = pd.DataFrame(d)
print(params_df)

# Forecast state level total GHG emissions

# First estimate state level GDP growth rates via the following model

# \ln{\left(\frac{y_{i,t}}{y_{i,t-1}}\right)} = \gamma_{0,t} + \gamma_{1}\ln{(y_{i,t-1})} + \gamma_{2}\ln{(y_{i,t-1}^{2})} + \theta_{i,t}

# Create dataframe with all data needed for this regression

gdp_filepath = 'C:/Users/User/Documents/Data/gdp_reg_data.csv'
gdp_data = pd.read_csv(gdp_filepath)
gdp_params = [[],[],[]]

for state in States:
    print(state)
    temp1 = []
    temp2 = []
    y = []
    for i in range(len(gdp_data)):
        if gdp_data.State[i] == state:
            temp1.append(gdp_data.ln_GDP[i])
            temp2.append(gdp_data.ln_GDP_2[i])
            y.append(gdp_data.Y[i])
    temp1 = pd.DataFrame(temp1)
    temp1.rename(columns = {0:'ln_GDP'}, inplace = True)
    temp2 = pd.DataFrame(temp2)
    temp2.rename(columns = {0:'ln_GDP_2'}, inplace = True)
    x = pd.concat([temp1, temp2], axis = 1)
    x = stats.add_constant(x)
    y = pd.DataFrame(y)
    y.rename(columns = {0:'Y'}, inplace = True)
    temp_gdp_mod = stats.OLS(y, x)
    temp_gdp_res = temp_gdp_mod.fit()
    print(temp_gdp_res.summary())
    gdp_params[0].append(temp_gdp_res.params[0])
    gdp_params[1].append(temp_gdp_res.params[1])
    gdp_params[2].append(temp_gdp_res.params[2])

dgdp = {'State':States, 'gamma_0':gdp_params[0], 'gamma_1':gdp_params[1], 'gamma_2':gdp_params[2]}
GDP_df = pd.DataFrame(dgdp)
GDP_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/gdp_forecast.txt', index = False)

# Unpack these ln-ratio estimates into actual forecasted data

gdp_data2 = GDP_df
gdp_forc = []

for state in States:
    fgdp = []
    fgdp1 = []
    fgdp2 = []
    for i in range(len(data)):
        if data['State'][i] == state:
            fgdp.append(data.GDP_per_capita[i])
            fgdp1.append(np.log(data.GDP_per_capita[i]))
            fgdp2.append(np.log(data.GDP_per_capita_2[i]))
    idx = gdp_data2[gdp_data2['State'] == state].index.values.astype(int)[0]
    for i in range(22,122):
        fgdp1.append((fgdp1[i-1] + gdp_data2['gamma_0'][idx] + gdp_data2['gamma_1'][idx]*fgdp1[i-1] + gdp_data2['gamma_2'][idx]*fgdp2[i-1]))
        fgdp.append(np.exp(fgdp1[i]))
        fgdp2.append(np.log(fgdp[i]**2))
    gdp_forc.append(fgdp)

gdpforcdic = {'State':States, 'forecasted_gdp_per_capita':gdp_forc}
GDP_df = pd.DataFrame(gdpforcdic)
GDP_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/gdp_forecast.txt', index = False)

# Second estimate future population densities

pop_den_forecast = []

for state in States:
    idx = params_df[params_df['State'] == state].index.values.astype(int)[0]
    temp = [(params_df['PD_int'][idx] + params_df['PD_slope'][idx]*i) for i in range(22,122)]
    pop_den_forecast.append(temp)

pdforc = {'State':States, 'forecasted_PD':pop_den_forecast}
PD_df = pd.DataFrame(pdforc)
PD_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/pd_forecast.txt', index = False)

# Third get raw estimates of state level forecasted emissions for the following cases:

forecasted_energy_ghg = []
beta = energy_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg.append(temp)

energy_forcdic = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg}
energy_forc_df = pd.DataFrame(energy_forcdic)
print(energy_forc_df)
energy_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST.txt', index = False)

forecasted_energy_ghg2 = []
beta2 = energy_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg2.append(temp)

energy_forcdic2 = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg2}
energy_forc_df2 = pd.DataFrame(energy_forcdic2)
print(energy_forc_df2)
energy_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_2.txt', index = False)

forecasted_energy_ghg_yr = []
beta_yr = energy_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg_yr.append(temp)

energy_forcdic_yr = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_yr}
energy_forc_df_yr = pd.DataFrame(energy_forcdic_yr)
print(energy_forc_df_yr)
energy_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_YR.txt', index = False)

forecasted_energy_ghg_yr2 = []
beta_yr2 = energy_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg_yr2.append(temp)

energy_forcdic_yr2 = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_yr2}
energy_forc_df_yr2 = pd.DataFrame(energy_forcdic_yr2)
print(energy_forc_df_yr2)
energy_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_YR_2.txt', index = False)

forecasted_energy_ghg_init = []
beta_init = energy_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg_init.append(temp)

energy_forcdic_init = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_init}
energy_forc_df_init = pd.DataFrame(energy_forcdic_init)
print(energy_forc_df_init)
energy_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_energy_ghg_init2 = []
beta_init2 = energy_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg_init2.append(temp)

energy_forcdic_init2 = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_init2}
energy_forc_df_init2 = pd.DataFrame(energy_forcdic_init2)
print(energy_forc_df_init2)
energy_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_energy_ghg_rand = []
beta_rand = energy_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg_rand.append(temp)

energy_forcdic_rand = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_rand}
energy_forc_df_rand = pd.DataFrame(energy_forcdic_rand)
print(energy_forc_df_rand)
energy_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_energy_ghg_rand2 = []
beta_rand2 = energy_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_energy_ghg_rand2.append(temp)

energy_forcdic_rand2 = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_rand2}
energy_forc_df_rand2 = pd.DataFrame(energy_forcdic_rand2)
print(energy_forc_df_rand2)
energy_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_energy_ghg_cpd = []
beta = energy_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_cpd.append(temp)

energy_forcdic_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_cpd}
energy_forc_df_cpd = pd.DataFrame(energy_forcdic_cpd)
print(energy_forc_df_cpd)
energy_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_NOPD.txt', index = False)

forecasted_energy_ghg2_cpd = []
beta2 = energy_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg2_cpd.append(temp)

energy_forcdic2_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg2_cpd}
energy_forc_df2_cpd = pd.DataFrame(energy_forcdic2_cpd)
print(energy_forc_df2_cpd)
energy_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_energy_ghg_yr_cpd = []
beta_yr = energy_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_yr_cpd.append(temp)

energy_forcdic_yr_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_yr_cpd}
energy_forc_df_yr_cpd = pd.DataFrame(energy_forcdic_yr_cpd)
print(energy_forc_df_yr_cpd)
energy_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_energy_ghg_yr2_cpd = []
beta_yr2 = energy_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_yr2_cpd.append(temp)

energy_forcdic_yr2_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_yr2_cpd}
energy_forc_df_yr2_cpd = pd.DataFrame(energy_forcdic_yr2_cpd)
print(energy_forc_df_yr2_cpd)
energy_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_energy_ghg_init_cpd = []
beta_init = energy_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_init_cpd.append(temp)

energy_forcdic_init_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_init_cpd}
energy_forc_df_init_cpd = pd.DataFrame(energy_forcdic_init_cpd)
print(energy_forc_df_init_cpd)
energy_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_energy_ghg_init2_cpd = []
beta_init2 = energy_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_init2_cpd.append(temp)

energy_forcdic_init2_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_init2_cpd}
energy_forc_df_init2_cpd = pd.DataFrame(energy_forcdic_init2_cpd)
print(energy_forc_df_init2_cpd)
energy_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_energy_ghg_rand_cpd = []
beta_rand = energy_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_rand_cpd.append(temp)

energy_forcdic_rand_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_rand_cpd}
energy_forc_df_rand_cpd = pd.DataFrame(energy_forcdic_rand_cpd)
print(energy_forc_df_rand_cpd)
energy_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_energy_ghg_rand2_cpd = []
beta_rand2 = energy_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_energy_ghg_rand2_cpd.append(temp)

energy_forcdic_rand2_cpd = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg_rand2_cpd}
energy_forc_df_rand2_cpd = pd.DataFrame(energy_forcdic_rand2_cpd)
print(energy_forc_df_rand2_cpd)
energy_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/ENERGY_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating energy production derived GHG emissions for the US

US_energy_GHG = np.zeros(22)
US_energy_GHG_fixed = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixed2 = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixedyr = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixedyr2 = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_init = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_init2 = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_rand = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_rand2 = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixed_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixed2_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixedyr_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_fixedyr2_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_init_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_init2_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_rand_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
US_energy_GHG_rand2_nopd = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
agg_energy = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Energy[i])
    agg_energy.append(temp)
    
for i in range(22):
    for j in range(len(agg_energy)):
        US_energy_GHG[i] += agg_energy[j][i]

for i in range(len(energy_forc_df)):
    for j in range(len(energy_forc_df['Forecasted_Energy_GHG'][0])):
        US_energy_GHG_fixed[j] += energy_forc_df['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixed2[j] += energy_forc_df2['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixedyr[j] += energy_forc_df_yr['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixedyr2[j] += energy_forc_df_yr2['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_init[j] += energy_forc_df_init['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_init2[j] += energy_forc_df_init2['Forecasted_Energy_GHG'][i][j]        
        US_energy_GHG_rand[j] += energy_forc_df_rand['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_rand2[j] += energy_forc_df_rand2['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixed_nopd[j] += energy_forc_df_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixed2_nopd[j] += energy_forc_df2_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixedyr_nopd[j] += energy_forc_df_yr_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_fixedyr2_nopd[j] += energy_forc_df_yr2_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_init_nopd[j] += energy_forc_df_init_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_init2_nopd[j] += energy_forc_df_init2_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_rand_nopd[j] += energy_forc_df_rand_cpd['Forecasted_Energy_GHG'][i][j]
        US_energy_GHG_rand2_nopd[j] += energy_forc_df_rand2_cpd['Forecasted_Energy_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [energy_fixed_results, energy_fixed_results2, energy_year_results, energy_year_results2, energy_results, energy_results2, energy_rand_results, energy_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Defining function for the tobit transformation

def tobit_transform(x,s):
    out = x*norm.cdf(x/s) + s*norm.pdf(x)
    return out

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_energy_GHG_init)):
    US_energy_GHG_fixed[i] = tobit_transform(US_energy_GHG_fixed[i], residual_std[0])
    US_energy_GHG_fixed2[i] = tobit_transform(US_energy_GHG_fixed2[i], residual_std[1])
    US_energy_GHG_fixedyr[i] = tobit_transform(US_energy_GHG_fixedyr[i], residual_std[2])
    US_energy_GHG_fixedyr2[i] = tobit_transform(US_energy_GHG_fixedyr2[i], residual_std[3])
    US_energy_GHG_init[i] = tobit_transform(US_energy_GHG_init[i], residual_std[4])
    US_energy_GHG_init2[i] = tobit_transform(US_energy_GHG_init2[i], residual_std[5])        
    US_energy_GHG_rand[i] = tobit_transform(US_energy_GHG_rand[i], residual_std[6])
    US_energy_GHG_rand2[i] = tobit_transform(US_energy_GHG_rand2[i], residual_std[7])        
    US_energy_GHG_fixed_nopd[i] = tobit_transform(US_energy_GHG_fixed_nopd[i], residual_std[0])
    US_energy_GHG_fixed2_nopd[i] = tobit_transform(US_energy_GHG_fixed2_nopd[i], residual_std[1])
    US_energy_GHG_fixedyr_nopd[i] = tobit_transform(US_energy_GHG_fixedyr_nopd[i], residual_std[2])
    US_energy_GHG_fixedyr2_nopd[i] = tobit_transform(US_energy_GHG_fixedyr2_nopd[i], residual_std[3])
    US_energy_GHG_init_nopd[i] = tobit_transform(US_energy_GHG_init_nopd[i], residual_std[4])
    US_energy_GHG_init2_nopd[i] = tobit_transform(US_energy_GHG_init2_nopd[i], residual_std[5])        
    US_energy_GHG_rand_nopd[i] = tobit_transform(US_energy_GHG_rand_nopd[i], residual_std[6])
    US_energy_GHG_rand2_nopd[i] = tobit_transform(US_energy_GHG_rand2_nopd[i], residual_std[7])

# Create a scatter plot of historical aggregated US GHG emissions from energy production
    
from pylab import rcParams
rcParams['figure.figsize'] = 8.5, 8.5
cm = plt.get_cmap('gist_rainbow')

plt.figure(0)
#plt.ylim(bottom = 250)
#plt.ylim(top = 525)
basis = [i for i in range(1990,2012)]
plt.plot(basis, US_energy_GHG, label = 'Historical Data')
 
# Add titles
plt.title('Historical Data', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/historical.eps')

# Create scatter plots for forecasted aggreagated US GHG emissions from energy production

plt.figure(1)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init, label = 'Baseline', color = cm(00))
plt.plot(basis, US_energy_GHG_init2, label = 'Baseline w/o cubic', color = cm(15))
plt.plot(basis, US_energy_GHG_fixedyr, label = 'Year only FE', color = cm(30))
plt.plot(basis, US_energy_GHG_fixedyr2, label = 'Year only FE w/o cubic', color = cm(45))
plt.plot(basis, US_energy_GHG_fixed, label = 'Fixed Effects', color = cm(60))
plt.plot(basis, US_energy_GHG_fixed2, label = 'Fixed Effects w/o cubic', color = cm(75))
plt.plot(basis, US_energy_GHG_rand, label = 'Random Effects', color = cm(90))
plt.plot(basis, US_energy_GHG_rand2, label = 'Random Effects w/o cubic', color = cm(105))
plt.plot(basis, US_energy_GHG_init_nopd, label = 'Baseline w/o PD', color = cm(120))
plt.plot(basis, US_energy_GHG_init2_nopd, label = 'Baseline w/o PD or cubic', color = cm(135))
plt.plot(basis, US_energy_GHG_fixedyr_nopd, label = 'Year only FE w/o PD', color = cm(150))
plt.plot(basis, US_energy_GHG_fixedyr2_nopd, label = 'Year only FE w/o PD or cubic', color = cm(165))
plt.plot(basis, US_energy_GHG_fixed_nopd, label = 'Fixed Effects w/o PD', color = cm(180))
plt.plot(basis, US_energy_GHG_fixed2_nopd, label = 'Fixed Effects w/o PD or cubic', color = cm(195))
plt.plot(basis, US_energy_GHG_rand_nopd, label = 'Random Effects w/o PD', color = cm(210))
plt.plot(basis, US_energy_GHG_rand2_nopd, label = 'Random Effects w/o PD or cubic', color = cm(225))

# Add legend
plt.legend(loc = 8, ncol = 2)
 
# Add titles
plt.title('Aggregated GHG Emissions from Energy Production\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/forecasts_all.eps')

plt.figure(2)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init, label = 'Baseline', color = cm(0))
plt.plot(basis, US_energy_GHG_fixedyr, label = 'Year only FE', color = cm(30))
plt.plot(basis, US_energy_GHG_fixed, label = 'Fixed Effects', color = cm(60))
plt.plot(basis, US_energy_GHG_rand, label = 'Random Effects', color = cm(90))
plt.plot(basis, US_energy_GHG_init_nopd, label = 'Baseline w/o PD', color = cm(120))
plt.plot(basis, US_energy_GHG_fixedyr_nopd, label = 'Year only FE w/o PD', color = cm(150))
plt.plot(basis, US_energy_GHG_fixed_nopd, label = 'Fixed Effects w/o PD', color = cm(180))
plt.plot(basis, US_energy_GHG_rand_nopd, label = 'Random Effects w/o PD', color = cm(210))

# Add legend
plt.legend(loc = 8, ncol = 1)
 
# Add titles
plt.title('Aggregated GHG Emissions from Energy Production\n(Models with cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/forecasts_cubic.eps')

plt.figure(3)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init2, label = 'Baseline w/o cubic', color = cm(15))
plt.plot(basis, US_energy_GHG_fixedyr2, label = 'Year only FE w/o cubic', color = cm(45))
plt.plot(basis, US_energy_GHG_fixed2, label = 'Fixed Effects w/o cubic', color = cm(75))
plt.plot(basis, US_energy_GHG_rand2, label = 'Random Effects w/o cubic', color = cm(105))
plt.plot(basis, US_energy_GHG_init2_nopd, label = 'Baseline w/o PD or cubic', color = cm(135))
plt.plot(basis, US_energy_GHG_fixedyr2_nopd, label = 'Year only FE w/o PD or cubic', color = cm(165))
plt.plot(basis, US_energy_GHG_fixed2_nopd, label = 'Fixed Effects w/o PD or cubic', color = cm(195))
plt.plot(basis, US_energy_GHG_rand2_nopd, label = 'Random Effects w/o PD or cubic', color = cm(225))

# Add legend
plt.legend(loc = 8, ncol = 1)
 
# Add titles
plt.title('Aggregated GHG Emissions from Energy Production\n(Models without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/forecasts_no_cubic.eps')

plt.figure(4)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init, label = 'Baseline', color = cm(0))
plt.plot(basis, US_energy_GHG_init2, label = 'Baseline w/o cubic', color = cm(30))
plt.plot(basis, US_energy_GHG_fixedyr, label = 'Year only FE', color = cm(60))
plt.plot(basis, US_energy_GHG_fixedyr2, label = 'Year only FE w/o cubic', color = cm(90))
plt.plot(basis, US_energy_GHG_fixed, label = 'Fixed Effects', color = cm(120))
plt.plot(basis, US_energy_GHG_fixed2, label = 'Fixed Effects w/o cubic', color = cm(150))
plt.plot(basis, US_energy_GHG_rand, label = 'Random Effects', color = cm(180))
plt.plot(basis, US_energy_GHG_rand2, label = 'Random Effects w/o cubic', color = cm(210))

# Add legend
plt.legend(loc = 8, ncol = 1)
 
# Add titles
plt.title('Aggregated GHG Emissions from Energy Production\n(Models with forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/forecasts_with_pop_den.eps')

plt.figure(5)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init_nopd, label = 'Baseline w/o PD', color = cm(15))
plt.plot(basis, US_energy_GHG_init2_nopd, label = 'Baseline w/o PD or cubic', color = cm(45))
plt.plot(basis, US_energy_GHG_fixedyr_nopd, label = 'Year only FE w/o PD', color = cm(75))
plt.plot(basis, US_energy_GHG_fixedyr2_nopd, label = 'Year only FE w/o PD or cubic', color = cm(105))
plt.plot(basis, US_energy_GHG_fixed_nopd, label = 'Fixed Effects w/o PD', color = cm(135))
plt.plot(basis, US_energy_GHG_fixed2_nopd, label = 'Fixed Effects w/o PD or cubic', color = cm(165))
plt.plot(basis, US_energy_GHG_rand_nopd, label = 'Random Effects w/o PD', color = cm(195))
plt.plot(basis, US_energy_GHG_rand2_nopd, label = 'Random Effects w/o PD or cubic', color = cm(225))

# Add legend
plt.legend(loc = 8, ncol = 1)
 
# Add titles
plt.title('Aggregated GHG Emissions from Energy Production\n(Models with constant population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/forecasts_without_pop_den.eps')

plt.figure(6)
for i in range(50):
    plt.plot(energy_forc_df['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nFixed Effects Model', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model.eps')

plt.figure(7)
for i in range(50):
    plt.plot(energy_forc_df2['Forecasted_Energy_GHG'][i])
    
# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nFixed Effects Model w/o cubic term', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model2.eps')
    
plt.figure(8)
for i in range(50):
    plt.plot(energy_forc_df_yr['Forecasted_Energy_GHG'][i])
    
# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nYear Fixed Effets Model', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model_yr.eps')

plt.figure(9)
for i in range(50):
    plt.plot(energy_forc_df_yr2['Forecasted_Energy_GHG'][i])
    
# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nYear Fixed Effects Model w/o cubic term', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model_yr2.eps')

plt.figure(10)
for i in range(50):
    plt.plot(energy_forc_df_init['Forecasted_Energy_GHG'][i])
    
# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nBaseline Model', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_init_model.eps')

plt.figure(11)
for i in range(50):
    plt.plot(energy_forc_df_init2['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nBaseline Model w/o cubic term', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_init_model2.eps')

plt.figure(12)
for i in range(50):
    plt.plot(energy_forc_df_rand['Forecasted_Energy_GHG'][i])
    
# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nRandom Effects Model', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_rand_model.eps')

plt.figure(13)
for i in range(50):
    plt.plot(energy_forc_df_rand2['Forecasted_Energy_GHG'][i])
    
# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\nRandom Effects Model w/o cubic term', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_rand_model2.eps')

plt.figure(14)
for i in range(50):
    plt.plot(energy_forc_df_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_fixed_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model_nopd.eps')

plt.figure(15)
for i in range(50):
    plt.plot(energy_forc_df2_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_fixed2_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model2_nopd.eps')
    
plt.figure(16)
for i in range(50):
    plt.plot(energy_forc_df_yr_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_fixedyr_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model_yr_nopd.eps')

plt.figure(17)
for i in range(50):
    plt.plot(energy_forc_df_yr2_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_fixed_yr2_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_fixed_model_yr2_nopd.eps')

plt.figure(18)
for i in range(50):
    plt.plot(energy_forc_df_init_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_init_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_init_model_nopd.eps')

plt.figure(19)
for i in range(50):
    plt.plot(energy_forc_df_init2_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_init2_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_init_model2_nopd.eps')

plt.figure(20)
for i in range(50):
    plt.plot(energy_forc_df_rand_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_rand_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_rand_model_nopd.eps')

plt.figure(21)
for i in range(50):
    plt.plot(energy_forc_df_rand2_cpd['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production\n(US_energy_GHG_rand2_nopd)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/all_states_rand_model2_nopd.eps')

# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_commercial_ghg = []
beta = commercial_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg.append(temp)

commercial_forcdic = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg}
commercial_forc_df = pd.DataFrame(commercial_forcdic)
print(commercial_forc_df)
commercial_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST.txt', index = False)

forecasted_commercial_ghg2 = []
beta2 = commercial_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg2.append(temp)

commercial_forcdic2 = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg2}
commercial_forc_df2 = pd.DataFrame(commercial_forcdic2)
print(commercial_forc_df2)
commercial_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_2.txt', index = False)

forecasted_commercial_ghg_yr = []
beta_yr = commercial_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg_yr.append(temp)

commercial_forcdic_yr = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_yr}
commercial_forc_df_yr = pd.DataFrame(commercial_forcdic_yr)
print(commercial_forc_df_yr)
commercial_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_YR.txt', index = False)

forecasted_commercial_ghg_yr2 = []
beta_yr2 = commercial_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg_yr2.append(temp)

commercial_forcdic_yr2 = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_yr2}
commercial_forc_df_yr2 = pd.DataFrame(commercial_forcdic_yr2)
print(commercial_forc_df_yr2)
commercial_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_YR_2.txt', index = False)

forecasted_commercial_ghg_init = []
beta_init = commercial_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg_init.append(temp)

commercial_forcdic_init = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_init}
commercial_forc_df_init = pd.DataFrame(commercial_forcdic_init)
print(commercial_forc_df_init)
commercial_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_commercial_ghg_init2 = []
beta_init2 = commercial_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg_init2.append(temp)

commercial_forcdic_init2 = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_init2}
commercial_forc_df_init2 = pd.DataFrame(commercial_forcdic_init2)
print(commercial_forc_df_init2)
commercial_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_commercial_ghg_rand = []
beta_rand = commercial_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg_rand.append(temp)

commercial_forcdic_rand = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_rand}
commercial_forc_df_rand = pd.DataFrame(commercial_forcdic_rand)
print(commercial_forc_df_rand)
commercial_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_commercial_ghg_rand2 = []
beta_rand2 = commercial_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_commercial_ghg_rand2.append(temp)

commercial_forcdic_rand2 = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_rand2}
commercial_forc_df_rand2 = pd.DataFrame(commercial_forcdic_rand2)
print(commercial_forc_df_rand2)
commercial_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_commercial_ghg_cpd = []
beta = commercial_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_cpd.append(temp)

commercial_forcdic_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_cpd}
commercial_forc_df_cpd = pd.DataFrame(commercial_forcdic_cpd)
print(commercial_forc_df_cpd)
commercial_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_NOPD.txt', index = False)

forecasted_commercial_ghg2_cpd = []
beta2 = commercial_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg2_cpd.append(temp)

commercial_forcdic2_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg2_cpd}
commercial_forc_df2_cpd = pd.DataFrame(commercial_forcdic2_cpd)
print(commercial_forc_df2_cpd)
commercial_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_commercial_ghg_yr_cpd = []
beta_yr = commercial_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_yr_cpd.append(temp)

commercial_forcdic_yr_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_yr_cpd}
commercial_forc_df_yr_cpd = pd.DataFrame(commercial_forcdic_yr_cpd)
print(commercial_forc_df_yr_cpd)
commercial_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_commercial_ghg_yr2_cpd = []
beta_yr2 = commercial_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_yr2_cpd.append(temp)

commercial_forcdic_yr2_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_yr2_cpd}
commercial_forc_df_yr2_cpd = pd.DataFrame(commercial_forcdic_yr2_cpd)
print(commercial_forc_df_yr2_cpd)
commercial_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_commercial_ghg_init_cpd = []
beta_init = commercial_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_init_cpd.append(temp)

commercial_forcdic_init_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_init_cpd}
commercial_forc_df_init_cpd = pd.DataFrame(commercial_forcdic_init_cpd)
print(commercial_forc_df_init_cpd)
commercial_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_commercial_ghg_init2_cpd = []
beta_init2 = commercial_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_init2_cpd.append(temp)

commercial_forcdic_init2_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_init2_cpd}
commercial_forc_df_init2_cpd = pd.DataFrame(commercial_forcdic_init2_cpd)
print(commercial_forc_df_init2_cpd)
commercial_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_commercial_ghg_rand_cpd = []
beta_rand = commercial_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_rand_cpd.append(temp)

commercial_forcdic_rand_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_rand_cpd}
commercial_forc_df_rand_cpd = pd.DataFrame(commercial_forcdic_rand_cpd)
print(commercial_forc_df_rand_cpd)
commercial_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_commercial_ghg_rand2_cpd = []
beta_rand2 = commercial_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_commercial_ghg_rand2_cpd.append(temp)

commercial_forcdic_rand2_cpd = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg_rand2_cpd}
commercial_forc_df_rand2_cpd = pd.DataFrame(commercial_forcdic_rand2_cpd)
print(commercial_forc_df_rand2_cpd)
commercial_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/commercial_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating commercial production derived GHG emissions for the US

US_commercial_GHG = np.zeros(22)
US_commercial_GHG_fixed = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixed2 = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixedyr = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixedyr2 = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_init = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_init2 = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_rand = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_rand2 = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixed_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixed2_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixedyr_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_fixedyr2_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_init_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_init2_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_rand_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
US_commercial_GHG_rand2_nopd = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
agg_commercial = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Commercial[i])
    agg_commercial.append(temp)
    
for i in range(22):
    for j in range(len(agg_commercial)):
        US_commercial_GHG[i] += agg_commercial[j][i]

for i in range(len(commercial_forc_df)):
    for j in range(len(commercial_forc_df['Forecasted_commercial_GHG'][0])):
        US_commercial_GHG_fixed[j] += commercial_forc_df['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixed2[j] += commercial_forc_df2['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixedyr[j] += commercial_forc_df_yr['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixedyr2[j] += commercial_forc_df_yr2['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_init[j] += commercial_forc_df_init['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_init2[j] += commercial_forc_df_init2['Forecasted_commercial_GHG'][i][j]        
        US_commercial_GHG_rand[j] += commercial_forc_df_rand['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_rand2[j] += commercial_forc_df_rand2['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixed_nopd[j] += commercial_forc_df_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixed2_nopd[j] += commercial_forc_df2_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixedyr_nopd[j] += commercial_forc_df_yr_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_fixedyr2_nopd[j] += commercial_forc_df_yr2_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_init_nopd[j] += commercial_forc_df_init_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_init2_nopd[j] += commercial_forc_df_init2_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_rand_nopd[j] += commercial_forc_df_rand_cpd['Forecasted_commercial_GHG'][i][j]
        US_commercial_GHG_rand2_nopd[j] += commercial_forc_df_rand2_cpd['Forecasted_commercial_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [commercial_fixed_results, commercial_fixed_results2, commercial_year_results, commercial_year_results2, commercial_results, commercial_results2, commercial_rand_results, commercial_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_commercial_GHG_init)):
    US_commercial_GHG_fixed[i] = tobit_transform(US_commercial_GHG_fixed[i], residual_std[0])
    US_commercial_GHG_fixed2[i] = tobit_transform(US_commercial_GHG_fixed2[i], residual_std[1])
    US_commercial_GHG_fixedyr[i] = tobit_transform(US_commercial_GHG_fixedyr[i], residual_std[2])
    US_commercial_GHG_fixedyr2[i] = tobit_transform(US_commercial_GHG_fixedyr2[i], residual_std[3])
    US_commercial_GHG_init[i] = tobit_transform(US_commercial_GHG_init[i], residual_std[4])
    US_commercial_GHG_init2[i] = tobit_transform(US_commercial_GHG_init2[i], residual_std[5])        
    US_commercial_GHG_rand[i] = tobit_transform(US_commercial_GHG_rand[i], residual_std[6])
    US_commercial_GHG_rand2[i] = tobit_transform(US_commercial_GHG_rand2[i], residual_std[7])        
    US_commercial_GHG_fixed_nopd[i] = tobit_transform(US_commercial_GHG_fixed_nopd[i], residual_std[0])
    US_commercial_GHG_fixed2_nopd[i] = tobit_transform(US_commercial_GHG_fixed2_nopd[i], residual_std[1])
    US_commercial_GHG_fixedyr_nopd[i] = tobit_transform(US_commercial_GHG_fixedyr_nopd[i], residual_std[2])
    US_commercial_GHG_fixedyr2_nopd[i] = tobit_transform(US_commercial_GHG_fixedyr2_nopd[i], residual_std[3])
    US_commercial_GHG_init_nopd[i] = tobit_transform(US_commercial_GHG_init_nopd[i], residual_std[4])
    US_commercial_GHG_init2_nopd[i] = tobit_transform(US_commercial_GHG_init2_nopd[i], residual_std[5])        
    US_commercial_GHG_rand_nopd[i] = tobit_transform(US_commercial_GHG_rand_nopd[i], residual_std[6])
    US_commercial_GHG_rand2_nopd[i] = tobit_transform(US_commercial_GHG_rand2_nopd[i], residual_std[7])

# Third get raw estimates of state level forecasted emissions for the following cases:

forecasted_residential_ghg = []
beta = residential_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg.append(temp)

residential_forcdic = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg}
residential_forc_df = pd.DataFrame(residential_forcdic)
print(residential_forc_df)
residential_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST.txt', index = False)

forecasted_residential_ghg2 = []
beta2 = residential_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg2.append(temp)

residential_forcdic2 = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg2}
residential_forc_df2 = pd.DataFrame(residential_forcdic2)
print(residential_forc_df2)
residential_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_2.txt', index = False)

forecasted_residential_ghg_yr = []
beta_yr = residential_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg_yr.append(temp)

residential_forcdic_yr = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_yr}
residential_forc_df_yr = pd.DataFrame(residential_forcdic_yr)
print(residential_forc_df_yr)
residential_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_YR.txt', index = False)

forecasted_residential_ghg_yr2 = []
beta_yr2 = residential_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg_yr2.append(temp)

residential_forcdic_yr2 = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_yr2}
residential_forc_df_yr2 = pd.DataFrame(residential_forcdic_yr2)
print(residential_forc_df_yr2)
residential_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_YR_2.txt', index = False)

forecasted_residential_ghg_init = []
beta_init = residential_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg_init.append(temp)

residential_forcdic_init = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_init}
residential_forc_df_init = pd.DataFrame(residential_forcdic_init)
print(residential_forc_df_init)
residential_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_residential_ghg_init2 = []
beta_init2 = residential_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg_init2.append(temp)

residential_forcdic_init2 = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_init2}
residential_forc_df_init2 = pd.DataFrame(residential_forcdic_init2)
print(residential_forc_df_init2)
residential_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_residential_ghg_rand = []
beta_rand = residential_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg_rand.append(temp)

residential_forcdic_rand = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_rand}
residential_forc_df_rand = pd.DataFrame(residential_forcdic_rand)
print(residential_forc_df_rand)
residential_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_residential_ghg_rand2 = []
beta_rand2 = residential_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_residential_ghg_rand2.append(temp)

residential_forcdic_rand2 = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_rand2}
residential_forc_df_rand2 = pd.DataFrame(residential_forcdic_rand2)
print(residential_forc_df_rand2)
residential_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_residential_ghg_cpd = []
beta = residential_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_cpd.append(temp)

residential_forcdic_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_cpd}
residential_forc_df_cpd = pd.DataFrame(residential_forcdic_cpd)
print(residential_forc_df_cpd)
residential_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_NOPD.txt', index = False)

forecasted_residential_ghg2_cpd = []
beta2 = residential_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg2_cpd.append(temp)

residential_forcdic2_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg2_cpd}
residential_forc_df2_cpd = pd.DataFrame(residential_forcdic2_cpd)
print(residential_forc_df2_cpd)
residential_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_residential_ghg_yr_cpd = []
beta_yr = residential_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_yr_cpd.append(temp)

residential_forcdic_yr_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_yr_cpd}
residential_forc_df_yr_cpd = pd.DataFrame(residential_forcdic_yr_cpd)
print(residential_forc_df_yr_cpd)
residential_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_residential_ghg_yr2_cpd = []
beta_yr2 = residential_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_yr2_cpd.append(temp)

residential_forcdic_yr2_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_yr2_cpd}
residential_forc_df_yr2_cpd = pd.DataFrame(residential_forcdic_yr2_cpd)
print(residential_forc_df_yr2_cpd)
residential_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_residential_ghg_init_cpd = []
beta_init = residential_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_init_cpd.append(temp)

residential_forcdic_init_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_init_cpd}
residential_forc_df_init_cpd = pd.DataFrame(residential_forcdic_init_cpd)
print(residential_forc_df_init_cpd)
residential_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_residential_ghg_init2_cpd = []
beta_init2 = residential_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_init2_cpd.append(temp)

residential_forcdic_init2_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_init2_cpd}
residential_forc_df_init2_cpd = pd.DataFrame(residential_forcdic_init2_cpd)
print(residential_forc_df_init2_cpd)
residential_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_residential_ghg_rand_cpd = []
beta_rand = residential_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_rand_cpd.append(temp)

residential_forcdic_rand_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_rand_cpd}
residential_forc_df_rand_cpd = pd.DataFrame(residential_forcdic_rand_cpd)
print(residential_forc_df_rand_cpd)
residential_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_residential_ghg_rand2_cpd = []
beta_rand2 = residential_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_residential_ghg_rand2_cpd.append(temp)

residential_forcdic_rand2_cpd = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg_rand2_cpd}
residential_forc_df_rand2_cpd = pd.DataFrame(residential_forcdic_rand2_cpd)
print(residential_forc_df_rand2_cpd)
residential_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/residential_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating residential production derived GHG emissions for the US

US_residential_GHG = np.zeros(22)
US_residential_GHG_fixed = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixed2 = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixedyr = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixedyr2 = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_init = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_init2 = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_rand = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_rand2 = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixed_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixed2_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixedyr_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_fixedyr2_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_init_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_init2_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_rand_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
US_residential_GHG_rand2_nopd = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
agg_residential = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Residential[i])
    agg_residential.append(temp)
    
for i in range(22):
    for j in range(len(agg_residential)):
        US_residential_GHG[i] += agg_residential[j][i]

for i in range(len(residential_forc_df)):
    for j in range(len(residential_forc_df['Forecasted_residential_GHG'][0])):
        US_residential_GHG_fixed[j] += residential_forc_df['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixed2[j] += residential_forc_df2['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixedyr[j] += residential_forc_df_yr['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixedyr2[j] += residential_forc_df_yr2['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_init[j] += residential_forc_df_init['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_init2[j] += residential_forc_df_init2['Forecasted_residential_GHG'][i][j]        
        US_residential_GHG_rand[j] += residential_forc_df_rand['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_rand2[j] += residential_forc_df_rand2['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixed_nopd[j] += residential_forc_df_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixed2_nopd[j] += residential_forc_df2_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixedyr_nopd[j] += residential_forc_df_yr_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_fixedyr2_nopd[j] += residential_forc_df_yr2_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_init_nopd[j] += residential_forc_df_init_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_init2_nopd[j] += residential_forc_df_init2_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_rand_nopd[j] += residential_forc_df_rand_cpd['Forecasted_residential_GHG'][i][j]
        US_residential_GHG_rand2_nopd[j] += residential_forc_df_rand2_cpd['Forecasted_residential_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [residential_fixed_results, residential_fixed_results2, residential_year_results, residential_year_results2, residential_results, residential_results2, residential_rand_results, residential_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_residential_GHG_init)):
    US_residential_GHG_fixed[i] = tobit_transform(US_residential_GHG_fixed[i], residual_std[0])
    US_residential_GHG_fixed2[i] = tobit_transform(US_residential_GHG_fixed2[i], residual_std[1])
    US_residential_GHG_fixedyr[i] = tobit_transform(US_residential_GHG_fixedyr[i], residual_std[2])
    US_residential_GHG_fixedyr2[i] = tobit_transform(US_residential_GHG_fixedyr2[i], residual_std[3])
    US_residential_GHG_init[i] = tobit_transform(US_residential_GHG_init[i], residual_std[4])
    US_residential_GHG_init2[i] = tobit_transform(US_residential_GHG_init2[i], residual_std[5])        
    US_residential_GHG_rand[i] = tobit_transform(US_residential_GHG_rand[i], residual_std[6])
    US_residential_GHG_rand2[i] = tobit_transform(US_residential_GHG_rand2[i], residual_std[7])        
    US_residential_GHG_fixed_nopd[i] = tobit_transform(US_residential_GHG_fixed_nopd[i], residual_std[0])
    US_residential_GHG_fixed2_nopd[i] = tobit_transform(US_residential_GHG_fixed2_nopd[i], residual_std[1])
    US_residential_GHG_fixedyr_nopd[i] = tobit_transform(US_residential_GHG_fixedyr_nopd[i], residual_std[2])
    US_residential_GHG_fixedyr2_nopd[i] = tobit_transform(US_residential_GHG_fixedyr2_nopd[i], residual_std[3])
    US_residential_GHG_init_nopd[i] = tobit_transform(US_residential_GHG_init_nopd[i], residual_std[4])
    US_residential_GHG_init2_nopd[i] = tobit_transform(US_residential_GHG_init2_nopd[i], residual_std[5])        
    US_residential_GHG_rand_nopd[i] = tobit_transform(US_residential_GHG_rand_nopd[i], residual_std[6])
    US_residential_GHG_rand2_nopd[i] = tobit_transform(US_residential_GHG_rand2_nopd[i], residual_std[7])

# Third get raw estimates of state level forecasted emissions for the following cases:

forecasted_industrial_ghg = []
beta = industrial_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg.append(temp)

industrial_forcdic = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg}
industrial_forc_df = pd.DataFrame(industrial_forcdic)
print(industrial_forc_df)
industrial_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST.txt', index = False)

forecasted_industrial_ghg2 = []
beta2 = industrial_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg2.append(temp)

industrial_forcdic2 = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg2}
industrial_forc_df2 = pd.DataFrame(industrial_forcdic2)
print(industrial_forc_df2)
industrial_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_2.txt', index = False)

forecasted_industrial_ghg_yr = []
beta_yr = industrial_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg_yr.append(temp)

industrial_forcdic_yr = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_yr}
industrial_forc_df_yr = pd.DataFrame(industrial_forcdic_yr)
print(industrial_forc_df_yr)
industrial_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_YR.txt', index = False)

forecasted_industrial_ghg_yr2 = []
beta_yr2 = industrial_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg_yr2.append(temp)

industrial_forcdic_yr2 = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_yr2}
industrial_forc_df_yr2 = pd.DataFrame(industrial_forcdic_yr2)
print(industrial_forc_df_yr2)
industrial_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_YR_2.txt', index = False)

forecasted_industrial_ghg_init = []
beta_init = industrial_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg_init.append(temp)

industrial_forcdic_init = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_init}
industrial_forc_df_init = pd.DataFrame(industrial_forcdic_init)
print(industrial_forc_df_init)
industrial_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_industrial_ghg_init2 = []
beta_init2 = industrial_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg_init2.append(temp)

industrial_forcdic_init2 = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_init2}
industrial_forc_df_init2 = pd.DataFrame(industrial_forcdic_init2)
print(industrial_forc_df_init2)
industrial_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_industrial_ghg_rand = []
beta_rand = industry_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg_rand.append(temp)

industrial_forcdic_rand = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_rand}
industrial_forc_df_rand = pd.DataFrame(industrial_forcdic_rand)
print(industrial_forc_df_rand)
industrial_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_industrial_ghg_rand2 = []
beta_rand2 = industry_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_industrial_ghg_rand2.append(temp)

industrial_forcdic_rand2 = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_rand2}
industrial_forc_df_rand2 = pd.DataFrame(industrial_forcdic_rand2)
print(industrial_forc_df_rand2)
industrial_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_industrial_ghg_cpd = []
beta = industrial_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_cpd.append(temp)

industrial_forcdic_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_cpd}
industrial_forc_df_cpd = pd.DataFrame(industrial_forcdic_cpd)
print(industrial_forc_df_cpd)
industrial_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_NOPD.txt', index = False)

forecasted_industrial_ghg2_cpd = []
beta2 = industrial_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg2_cpd.append(temp)

industrial_forcdic2_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg2_cpd}
industrial_forc_df2_cpd = pd.DataFrame(industrial_forcdic2_cpd)
print(industrial_forc_df2_cpd)
industrial_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_industrial_ghg_yr_cpd = []
beta_yr = industrial_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_yr_cpd.append(temp)

industrial_forcdic_yr_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_yr_cpd}
industrial_forc_df_yr_cpd = pd.DataFrame(industrial_forcdic_yr_cpd)
print(industrial_forc_df_yr_cpd)
industrial_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_industrial_ghg_yr2_cpd = []
beta_yr2 = industrial_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_yr2_cpd.append(temp)

industrial_forcdic_yr2_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_yr2_cpd}
industrial_forc_df_yr2_cpd = pd.DataFrame(industrial_forcdic_yr2_cpd)
print(industrial_forc_df_yr2_cpd)
industrial_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_industrial_ghg_init_cpd = []
beta_init = industrial_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_init_cpd.append(temp)

industrial_forcdic_init_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_init_cpd}
industrial_forc_df_init_cpd = pd.DataFrame(industrial_forcdic_init_cpd)
print(industrial_forc_df_init_cpd)
industrial_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_industrial_ghg_init2_cpd = []
beta_init2 = industrial_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_init2_cpd.append(temp)

industrial_forcdic_init2_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_init2_cpd}
industrial_forc_df_init2_cpd = pd.DataFrame(industrial_forcdic_init2_cpd)
print(industrial_forc_df_init2_cpd)
industrial_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_industrial_ghg_rand_cpd = []
beta_rand = industry_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_rand_cpd.append(temp)

industrial_forcdic_rand_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_rand_cpd}
industrial_forc_df_rand_cpd = pd.DataFrame(industrial_forcdic_rand_cpd)
print(industrial_forc_df_rand_cpd)
industrial_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_industrial_ghg_rand2_cpd = []
beta_rand2 = industry_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_industrial_ghg_rand2_cpd.append(temp)

industrial_forcdic_rand2_cpd = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg_rand2_cpd}
industrial_forc_df_rand2_cpd = pd.DataFrame(industrial_forcdic_rand2_cpd)
print(industrial_forc_df_rand2_cpd)
industrial_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/industrial_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating industrial production derived GHG emissions for the US

US_industrial_GHG = np.zeros(22)
US_industrial_GHG_fixed = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixed2 = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixedyr = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixedyr2 = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_init = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_init2 = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_rand = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_rand2 = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixed_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixed2_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixedyr_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_fixedyr2_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_init_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_init2_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_rand_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
US_industrial_GHG_rand2_nopd = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
agg_industrial = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Industrial[i])
    agg_industrial.append(temp)
    
for i in range(22):
    for j in range(len(agg_industrial)):
        US_industrial_GHG[i] += agg_industrial[j][i]

for i in range(len(industrial_forc_df)):
    for j in range(len(industrial_forc_df['Forecasted_industrial_GHG'][0])):
        US_industrial_GHG_fixed[j] += industrial_forc_df['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixed2[j] += industrial_forc_df2['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixedyr[j] += industrial_forc_df_yr['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixedyr2[j] += industrial_forc_df_yr2['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_init[j] += industrial_forc_df_init['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_init2[j] += industrial_forc_df_init2['Forecasted_industrial_GHG'][i][j]        
        US_industrial_GHG_rand[j] += industrial_forc_df_rand['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_rand2[j] += industrial_forc_df_rand2['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixed_nopd[j] += industrial_forc_df_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixed2_nopd[j] += industrial_forc_df2_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixedyr_nopd[j] += industrial_forc_df_yr_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_fixedyr2_nopd[j] += industrial_forc_df_yr2_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_init_nopd[j] += industrial_forc_df_init_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_init2_nopd[j] += industrial_forc_df_init2_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_rand_nopd[j] += industrial_forc_df_rand_cpd['Forecasted_industrial_GHG'][i][j]
        US_industrial_GHG_rand2_nopd[j] += industrial_forc_df_rand2_cpd['Forecasted_industrial_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [industrial_fixed_results, industrial_fixed_results2, industrial_year_results, industrial_year_results2, industrial_results, industrial_results2, industry_rand_results, industry_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_industrial_GHG_init)):
    US_industrial_GHG_fixed[i] = tobit_transform(US_industrial_GHG_fixed[i], residual_std[0])
    US_industrial_GHG_fixed2[i] = tobit_transform(US_industrial_GHG_fixed2[i], residual_std[1])
    US_industrial_GHG_fixedyr[i] = tobit_transform(US_industrial_GHG_fixedyr[i], residual_std[2])
    US_industrial_GHG_fixedyr2[i] = tobit_transform(US_industrial_GHG_fixedyr2[i], residual_std[3])
    US_industrial_GHG_init[i] = tobit_transform(US_industrial_GHG_init[i], residual_std[4])
    US_industrial_GHG_init2[i] = tobit_transform(US_industrial_GHG_init2[i], residual_std[5])        
    US_industrial_GHG_rand[i] = tobit_transform(US_industrial_GHG_rand[i], residual_std[6])
    US_industrial_GHG_rand2[i] = tobit_transform(US_industrial_GHG_rand2[i], residual_std[7])        
    US_industrial_GHG_fixed_nopd[i] = tobit_transform(US_industrial_GHG_fixed_nopd[i], residual_std[0])
    US_industrial_GHG_fixed2_nopd[i] = tobit_transform(US_industrial_GHG_fixed2_nopd[i], residual_std[1])
    US_industrial_GHG_fixedyr_nopd[i] = tobit_transform(US_industrial_GHG_fixedyr_nopd[i], residual_std[2])
    US_industrial_GHG_fixedyr2_nopd[i] = tobit_transform(US_industrial_GHG_fixedyr2_nopd[i], residual_std[3])
    US_industrial_GHG_init_nopd[i] = tobit_transform(US_industrial_GHG_init_nopd[i], residual_std[4])
    US_industrial_GHG_init2_nopd[i] = tobit_transform(US_industrial_GHG_init2_nopd[i], residual_std[5])        
    US_industrial_GHG_rand_nopd[i] = tobit_transform(US_industrial_GHG_rand_nopd[i], residual_std[6])
    US_industrial_GHG_rand2_nopd[i] = tobit_transform(US_industrial_GHG_rand2_nopd[i], residual_std[7])

# Third get raw estimates of state level forecasted emissions for the following cases:

forecasted_transportation_ghg = []
beta = transportation_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg.append(temp)

transportation_forcdic = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg}
transportation_forc_df = pd.DataFrame(transportation_forcdic)
print(transportation_forc_df)
transportation_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST.txt', index = False)

forecasted_transportation_ghg2 = []
beta2 = transportation_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg2.append(temp)

transportation_forcdic2 = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg2}
transportation_forc_df2 = pd.DataFrame(transportation_forcdic2)
print(transportation_forc_df2)
transportation_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_2.txt', index = False)

forecasted_transportation_ghg_yr = []
beta_yr = transportation_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg_yr.append(temp)

transportation_forcdic_yr = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_yr}
transportation_forc_df_yr = pd.DataFrame(transportation_forcdic_yr)
print(transportation_forc_df_yr)
transportation_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_YR.txt', index = False)

forecasted_transportation_ghg_yr2 = []
beta_yr2 = transportation_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg_yr2.append(temp)

transportation_forcdic_yr2 = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_yr2}
transportation_forc_df_yr2 = pd.DataFrame(transportation_forcdic_yr2)
print(transportation_forc_df_yr2)
transportation_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_YR_2.txt', index = False)

forecasted_transportation_ghg_init = []
beta_init = transportation_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg_init.append(temp)

transportation_forcdic_init = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_init}
transportation_forc_df_init = pd.DataFrame(transportation_forcdic_init)
print(transportation_forc_df_init)
transportation_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_transportation_ghg_init2 = []
beta_init2 = transportation_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg_init2.append(temp)

transportation_forcdic_init2 = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_init2}
transportation_forc_df_init2 = pd.DataFrame(transportation_forcdic_init2)
print(transportation_forc_df_init2)
transportation_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_transportation_ghg_rand = []
beta_rand = transportation_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg_rand.append(temp)

transportation_forcdic_rand = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_rand}
transportation_forc_df_rand = pd.DataFrame(transportation_forcdic_rand)
print(transportation_forc_df_rand)
transportation_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_transportation_ghg_rand2 = []
beta_rand2 = transportation_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_transportation_ghg_rand2.append(temp)

transportation_forcdic_rand2 = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_rand2}
transportation_forc_df_rand2 = pd.DataFrame(transportation_forcdic_rand2)
print(transportation_forc_df_rand2)
transportation_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_transportation_ghg_cpd = []
beta = transportation_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_cpd.append(temp)

transportation_forcdic_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_cpd}
transportation_forc_df_cpd = pd.DataFrame(transportation_forcdic_cpd)
print(transportation_forc_df_cpd)
transportation_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_NOPD.txt', index = False)

forecasted_transportation_ghg2_cpd = []
beta2 = transportation_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg2_cpd.append(temp)

transportation_forcdic2_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg2_cpd}
transportation_forc_df2_cpd = pd.DataFrame(transportation_forcdic2_cpd)
print(transportation_forc_df2_cpd)
transportation_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_transportation_ghg_yr_cpd = []
beta_yr = transportation_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_yr_cpd.append(temp)

transportation_forcdic_yr_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_yr_cpd}
transportation_forc_df_yr_cpd = pd.DataFrame(transportation_forcdic_yr_cpd)
print(transportation_forc_df_yr_cpd)
transportation_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_transportation_ghg_yr2_cpd = []
beta_yr2 = transportation_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_yr2_cpd.append(temp)

transportation_forcdic_yr2_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_yr2_cpd}
transportation_forc_df_yr2_cpd = pd.DataFrame(transportation_forcdic_yr2_cpd)
print(transportation_forc_df_yr2_cpd)
transportation_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_transportation_ghg_init_cpd = []
beta_init = transportation_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_init_cpd.append(temp)

transportation_forcdic_init_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_init_cpd}
transportation_forc_df_init_cpd = pd.DataFrame(transportation_forcdic_init_cpd)
print(transportation_forc_df_init_cpd)
transportation_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_transportation_ghg_init2_cpd = []
beta_init2 = transportation_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_init2_cpd.append(temp)

transportation_forcdic_init2_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_init2_cpd}
transportation_forc_df_init2_cpd = pd.DataFrame(transportation_forcdic_init2_cpd)
print(transportation_forc_df_init2_cpd)
transportation_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_transportation_ghg_rand_cpd = []
beta_rand = transportation_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_rand_cpd.append(temp)

transportation_forcdic_rand_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_rand_cpd}
transportation_forc_df_rand_cpd = pd.DataFrame(transportation_forcdic_rand_cpd)
print(transportation_forc_df_rand_cpd)
transportation_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_transportation_ghg_rand2_cpd = []
beta_rand2 = transportation_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_transportation_ghg_rand2_cpd.append(temp)

transportation_forcdic_rand2_cpd = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg_rand2_cpd}
transportation_forc_df_rand2_cpd = pd.DataFrame(transportation_forcdic_rand2_cpd)
print(transportation_forc_df_rand2_cpd)
transportation_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/transportation_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating transportation production derived GHG emissions for the US

US_transportation_GHG = np.zeros(22)
US_transportation_GHG_fixed = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixed2 = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixedyr = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixedyr2 = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_init = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_init2 = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_rand = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_rand2 = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixed_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixed2_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixedyr_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_fixedyr2_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_init_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_init2_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_rand_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
US_transportation_GHG_rand2_nopd = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
agg_transportation = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Transportation[i])
    agg_transportation.append(temp)
    
for i in range(22):
    for j in range(len(agg_transportation)):
        US_transportation_GHG[i] += agg_transportation[j][i]

for i in range(len(transportation_forc_df)):
    for j in range(len(transportation_forc_df['Forecasted_transportation_GHG'][0])):
        US_transportation_GHG_fixed[j] += transportation_forc_df['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixed2[j] += transportation_forc_df2['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixedyr[j] += transportation_forc_df_yr['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixedyr2[j] += transportation_forc_df_yr2['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_init[j] += transportation_forc_df_init['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_init2[j] += transportation_forc_df_init2['Forecasted_transportation_GHG'][i][j]        
        US_transportation_GHG_rand[j] += transportation_forc_df_rand['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_rand2[j] += transportation_forc_df_rand2['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixed_nopd[j] += transportation_forc_df_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixed2_nopd[j] += transportation_forc_df2_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixedyr_nopd[j] += transportation_forc_df_yr_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_fixedyr2_nopd[j] += transportation_forc_df_yr2_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_init_nopd[j] += transportation_forc_df_init_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_init2_nopd[j] += transportation_forc_df_init2_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_rand_nopd[j] += transportation_forc_df_rand_cpd['Forecasted_transportation_GHG'][i][j]
        US_transportation_GHG_rand2_nopd[j] += transportation_forc_df_rand2_cpd['Forecasted_transportation_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [transportation_fixed_results, transportation_fixed_results2, transportation_year_results, transportation_year_results2, transportation_results, transportation_results2, transportation_rand_results, transportation_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_transportation_GHG_init)):
    US_transportation_GHG_fixed[i] = tobit_transform(US_transportation_GHG_fixed[i], residual_std[0])
    US_transportation_GHG_fixed2[i] = tobit_transform(US_transportation_GHG_fixed2[i], residual_std[1])
    US_transportation_GHG_fixedyr[i] = tobit_transform(US_transportation_GHG_fixedyr[i], residual_std[2])
    US_transportation_GHG_fixedyr2[i] = tobit_transform(US_transportation_GHG_fixedyr2[i], residual_std[3])
    US_transportation_GHG_init[i] = tobit_transform(US_transportation_GHG_init[i], residual_std[4])
    US_transportation_GHG_init2[i] = tobit_transform(US_transportation_GHG_init2[i], residual_std[5])        
    US_transportation_GHG_rand[i] = tobit_transform(US_transportation_GHG_rand[i], residual_std[6])
    US_transportation_GHG_rand2[i] = tobit_transform(US_transportation_GHG_rand2[i], residual_std[7])        
    US_transportation_GHG_fixed_nopd[i] = tobit_transform(US_transportation_GHG_fixed_nopd[i], residual_std[0])
    US_transportation_GHG_fixed2_nopd[i] = tobit_transform(US_transportation_GHG_fixed2_nopd[i], residual_std[1])
    US_transportation_GHG_fixedyr_nopd[i] = tobit_transform(US_transportation_GHG_fixedyr_nopd[i], residual_std[2])
    US_transportation_GHG_fixedyr2_nopd[i] = tobit_transform(US_transportation_GHG_fixedyr2_nopd[i], residual_std[3])
    US_transportation_GHG_init_nopd[i] = tobit_transform(US_transportation_GHG_init_nopd[i], residual_std[4])
    US_transportation_GHG_init2_nopd[i] = tobit_transform(US_transportation_GHG_init2_nopd[i], residual_std[5])        
    US_transportation_GHG_rand_nopd[i] = tobit_transform(US_transportation_GHG_rand_nopd[i], residual_std[6])
    US_transportation_GHG_rand2_nopd[i] = tobit_transform(US_transportation_GHG_rand2_nopd[i], residual_std[7])

# Third get raw estimates of state level forecasted emissions for the following cases:

forecasted_electric_ghg = []
beta = electric_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg.append(temp)

electric_forcdic = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg}
electric_forc_df = pd.DataFrame(electric_forcdic)
print(electric_forc_df)
electric_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST.txt', index = False)

forecasted_electric_ghg2 = []
beta2 = electric_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg2.append(temp)

electric_forcdic2 = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg2}
electric_forc_df2 = pd.DataFrame(electric_forcdic2)
print(electric_forc_df2)
electric_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_2.txt', index = False)

forecasted_electric_ghg_yr = []
beta_yr = electric_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg_yr.append(temp)

electric_forcdic_yr = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_yr}
electric_forc_df_yr = pd.DataFrame(electric_forcdic_yr)
print(electric_forc_df_yr)
electric_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_YR.txt', index = False)

forecasted_electric_ghg_yr2 = []
beta_yr2 = electric_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg_yr2.append(temp)

electric_forcdic_yr2 = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_yr2}
electric_forc_df_yr2 = pd.DataFrame(electric_forcdic_yr2)
print(electric_forc_df_yr2)
electric_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_YR_2.txt', index = False)

forecasted_electric_ghg_init = []
beta_init = electric_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg_init.append(temp)

electric_forcdic_init = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_init}
electric_forc_df_init = pd.DataFrame(electric_forcdic_init)
print(electric_forc_df_init)
electric_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_electric_ghg_init2 = []
beta_init2 = electric_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg_init2.append(temp)

electric_forcdic_init2 = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_init2}
electric_forc_df_init2 = pd.DataFrame(electric_forcdic_init2)
print(electric_forc_df_init2)
electric_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_electric_ghg_rand = []
beta_rand = electric_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg_rand.append(temp)

electric_forcdic_rand = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_rand}
electric_forc_df_rand = pd.DataFrame(electric_forcdic_rand)
print(electric_forc_df_rand)
electric_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_electric_ghg_rand2 = []
beta_rand2 = electric_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_electric_ghg_rand2.append(temp)

electric_forcdic_rand2 = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_rand2}
electric_forc_df_rand2 = pd.DataFrame(electric_forcdic_rand2)
print(electric_forc_df_rand2)
electric_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_electric_ghg_cpd = []
beta = electric_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_cpd.append(temp)

electric_forcdic_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_cpd}
electric_forc_df_cpd = pd.DataFrame(electric_forcdic_cpd)
print(electric_forc_df_cpd)
electric_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_NOPD.txt', index = False)

forecasted_electric_ghg2_cpd = []
beta2 = electric_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg2_cpd.append(temp)

electric_forcdic2_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg2_cpd}
electric_forc_df2_cpd = pd.DataFrame(electric_forcdic2_cpd)
print(electric_forc_df2_cpd)
electric_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_electric_ghg_yr_cpd = []
beta_yr = electric_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_yr_cpd.append(temp)

electric_forcdic_yr_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_yr_cpd}
electric_forc_df_yr_cpd = pd.DataFrame(electric_forcdic_yr_cpd)
print(electric_forc_df_yr_cpd)
electric_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_electric_ghg_yr2_cpd = []
beta_yr2 = electric_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_yr2_cpd.append(temp)

electric_forcdic_yr2_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_yr2_cpd}
electric_forc_df_yr2_cpd = pd.DataFrame(electric_forcdic_yr2_cpd)
print(electric_forc_df_yr2_cpd)
electric_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_electric_ghg_init_cpd = []
beta_init = electric_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_init_cpd.append(temp)

electric_forcdic_init_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_init_cpd}
electric_forc_df_init_cpd = pd.DataFrame(electric_forcdic_init_cpd)
print(electric_forc_df_init_cpd)
electric_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_electric_ghg_init2_cpd = []
beta_init2 = electric_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_init2_cpd.append(temp)

electric_forcdic_init2_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_init2_cpd}
electric_forc_df_init2_cpd = pd.DataFrame(electric_forcdic_init2_cpd)
print(electric_forc_df_init2_cpd)
electric_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_electric_ghg_rand_cpd = []
beta_rand = electric_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_rand_cpd.append(temp)

electric_forcdic_rand_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_rand_cpd}
electric_forc_df_rand_cpd = pd.DataFrame(electric_forcdic_rand_cpd)
print(electric_forc_df_rand_cpd)
electric_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_electric_ghg_rand2_cpd = []
beta_rand2 = electric_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_electric_ghg_rand2_cpd.append(temp)

electric_forcdic_rand2_cpd = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg_rand2_cpd}
electric_forc_df_rand2_cpd = pd.DataFrame(electric_forcdic_rand2_cpd)
print(electric_forc_df_rand2_cpd)
electric_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/electric_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating electric production derived GHG emissions for the US

US_electric_GHG = np.zeros(22)
US_electric_GHG_fixed = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixed2 = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixedyr = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixedyr2 = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_init = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_init2 = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_rand = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_rand2 = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixed_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixed2_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixedyr_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_fixedyr2_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_init_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_init2_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_rand_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
US_electric_GHG_rand2_nopd = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
agg_electric = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Electric_Power[i])
    agg_electric.append(temp)
    
for i in range(22):
    for j in range(len(agg_electric)):
        US_electric_GHG[i] += agg_electric[j][i]

for i in range(len(electric_forc_df)):
    for j in range(len(electric_forc_df['Forecasted_electric_GHG'][0])):
        US_electric_GHG_fixed[j] += electric_forc_df['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixed2[j] += electric_forc_df2['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixedyr[j] += electric_forc_df_yr['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixedyr2[j] += electric_forc_df_yr2['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_init[j] += electric_forc_df_init['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_init2[j] += electric_forc_df_init2['Forecasted_electric_GHG'][i][j]        
        US_electric_GHG_rand[j] += electric_forc_df_rand['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_rand2[j] += electric_forc_df_rand2['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixed_nopd[j] += electric_forc_df_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixed2_nopd[j] += electric_forc_df2_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixedyr_nopd[j] += electric_forc_df_yr_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_fixedyr2_nopd[j] += electric_forc_df_yr2_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_init_nopd[j] += electric_forc_df_init_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_init2_nopd[j] += electric_forc_df_init2_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_rand_nopd[j] += electric_forc_df_rand_cpd['Forecasted_electric_GHG'][i][j]
        US_electric_GHG_rand2_nopd[j] += electric_forc_df_rand2_cpd['Forecasted_electric_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [electric_fixed_results, electric_fixed_results2, electric_year_results, electric_year_results2, electric_results, electric_results2, electric_rand_results, electric_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_electric_GHG_init)):
    US_electric_GHG_fixed[i] = tobit_transform(US_electric_GHG_fixed[i], residual_std[0])
    US_electric_GHG_fixed2[i] = tobit_transform(US_electric_GHG_fixed2[i], residual_std[1])
    US_electric_GHG_fixedyr[i] = tobit_transform(US_electric_GHG_fixedyr[i], residual_std[2])
    US_electric_GHG_fixedyr2[i] = tobit_transform(US_electric_GHG_fixedyr2[i], residual_std[3])
    US_electric_GHG_init[i] = tobit_transform(US_electric_GHG_init[i], residual_std[4])
    US_electric_GHG_init2[i] = tobit_transform(US_electric_GHG_init2[i], residual_std[5])        
    US_electric_GHG_rand[i] = tobit_transform(US_electric_GHG_rand[i], residual_std[6])
    US_electric_GHG_rand2[i] = tobit_transform(US_electric_GHG_rand2[i], residual_std[7])        
    US_electric_GHG_fixed_nopd[i] = tobit_transform(US_electric_GHG_fixed_nopd[i], residual_std[0])
    US_electric_GHG_fixed2_nopd[i] = tobit_transform(US_electric_GHG_fixed2_nopd[i], residual_std[1])
    US_electric_GHG_fixedyr_nopd[i] = tobit_transform(US_electric_GHG_fixedyr_nopd[i], residual_std[2])
    US_electric_GHG_fixedyr2_nopd[i] = tobit_transform(US_electric_GHG_fixedyr2_nopd[i], residual_std[3])
    US_electric_GHG_init_nopd[i] = tobit_transform(US_electric_GHG_init_nopd[i], residual_std[4])
    US_electric_GHG_init2_nopd[i] = tobit_transform(US_electric_GHG_init2_nopd[i], residual_std[5])        
    US_electric_GHG_rand_nopd[i] = tobit_transform(US_electric_GHG_rand_nopd[i], residual_std[6])
    US_electric_GHG_rand2_nopd[i] = tobit_transform(US_electric_GHG_rand2_nopd[i], residual_std[7])

# Third get raw estimates of state level forecasted emissions for the following cases:

forecasted_fugitive_ghg = []
beta = fugitive_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg.append(temp)

fugitive_forcdic = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg}
fugitive_forc_df = pd.DataFrame(fugitive_forcdic)
print(fugitive_forc_df)
fugitive_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST.txt', index = False)

forecasted_fugitive_ghg2 = []
beta2 = fugitive_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg2.append(temp)

fugitive_forcdic2 = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg2}
fugitive_forc_df2 = pd.DataFrame(fugitive_forcdic2)
print(fugitive_forc_df2)
fugitive_forc_df2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_2.txt', index = False)

forecasted_fugitive_ghg_yr = []
beta_yr = fugitive_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg_yr.append(temp)

fugitive_forcdic_yr = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_yr}
fugitive_forc_df_yr = pd.DataFrame(fugitive_forcdic_yr)
print(fugitive_forc_df_yr)
fugitive_forc_df_yr.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_YR.txt', index = False)

forecasted_fugitive_ghg_yr2 = []
beta_yr2 = fugitive_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg_yr2.append(temp)

fugitive_forcdic_yr2 = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_yr2}
fugitive_forc_df_yr2 = pd.DataFrame(fugitive_forcdic_yr2)
print(fugitive_forc_df_yr2)
fugitive_forc_df_yr2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_YR_2.txt', index = False)

forecasted_fugitive_ghg_init = []
beta_init = fugitive_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg_init.append(temp)

fugitive_forcdic_init = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_init}
fugitive_forc_df_init = pd.DataFrame(fugitive_forcdic_init)
print(fugitive_forc_df_init)
fugitive_forc_df_init.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_BASELINE.txt', index = False)

forecasted_fugitive_ghg_init2 = []
beta_init2 = fugitive_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg_init2.append(temp)

fugitive_forcdic_init2 = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_init2}
fugitive_forc_df_init2 = pd.DataFrame(fugitive_forcdic_init2)
print(fugitive_forc_df_init2)
fugitive_forc_df_init2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_BASELINE_2.txt', index = False)

forecasted_fugitive_ghg_rand = []
beta_rand = fugitive_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg_rand.append(temp)

fugitive_forcdic_rand = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_rand}
fugitive_forc_df_rand = pd.DataFrame(fugitive_forcdic_rand)
print(fugitive_forc_df_rand)
fugitive_forc_df_rand.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_RANDOM.txt', index = False)

forecasted_fugitive_ghg_rand2 = []
beta_rand2 = fugitive_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][i]))
    forecasted_fugitive_ghg_rand2.append(temp)

fugitive_forcdic_rand2 = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_rand2}
fugitive_forc_df_rand2 = pd.DataFrame(fugitive_forcdic_rand2)
print(fugitive_forc_df_rand2)
fugitive_forc_df_rand2.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_RANDOM_2.txt', index = False)

# Forecasts without population density forecasts

forecasted_fugitive_ghg_cpd = []
beta = fugitive_fixed_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta[state]))
        except:
            temp.append(max(0,beta['const'] + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_cpd.append(temp)

fugitive_forcdic_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_cpd}
fugitive_forc_df_cpd = pd.DataFrame(fugitive_forcdic_cpd)
print(fugitive_forc_df_cpd)
fugitive_forc_df_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_NOPD.txt', index = False)

forecasted_fugitive_ghg2_cpd = []
beta2 = fugitive_fixed_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta2[state]))
        except:
            temp.append(max(0,beta2['const'] + beta2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg2_cpd.append(temp)

fugitive_forcdic2_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg2_cpd}
fugitive_forc_df2_cpd = pd.DataFrame(fugitive_forcdic2_cpd)
print(fugitive_forc_df2_cpd)
fugitive_forc_df2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_2_NOPD.txt', index = False)

forecasted_fugitive_ghg_yr_cpd = []
beta_yr = fugitive_year_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr[state]))
        except:
            temp.append(max(0,beta_yr['const'] + beta_yr['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_yr['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_yr_cpd.append(temp)

fugitive_forcdic_yr_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_yr_cpd}
fugitive_forc_df_yr_cpd = pd.DataFrame(fugitive_forcdic_yr_cpd)
print(fugitive_forc_df_yr_cpd)
fugitive_forc_df_yr_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_YR_NOPD.txt', index = False)

forecasted_fugitive_ghg_yr2_cpd = []
beta_yr2 = fugitive_year_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_yr2[state]))
        except:
            temp.append(max(0,beta_yr2['const'] + beta_yr2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_yr2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_yr2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_yr2_cpd.append(temp)

fugitive_forcdic_yr2_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_yr2_cpd}
fugitive_forc_df_yr2_cpd = pd.DataFrame(fugitive_forcdic_yr2_cpd)
print(fugitive_forc_df_yr2_cpd)
fugitive_forc_df_yr2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_YR_2_NOPD.txt', index = False)

forecasted_fugitive_ghg_init_cpd = []
beta_init = fugitive_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init[state]))
        except:
            temp.append(max(0,beta_init['const'] + beta_init['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_init['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_init_cpd.append(temp)

fugitive_forcdic_init_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_init_cpd}
fugitive_forc_df_init_cpd = pd.DataFrame(fugitive_forcdic_init_cpd)
print(fugitive_forc_df_init_cpd)
fugitive_forc_df_init_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_BASELINE_NOPD.txt', index = False)

forecasted_fugitive_ghg_init2_cpd = []
beta_init2 = fugitive_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_init2[state]))
        except:
            temp.append(max(0,beta_init2['const'] + beta_init2['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_init2['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_init2['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_init2_cpd.append(temp)

fugitive_forcdic_init2_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_init2_cpd}
fugitive_forc_df_init2_cpd = pd.DataFrame(fugitive_forcdic_init2_cpd)
print(fugitive_forc_df_init2_cpd)
fugitive_forc_df_init2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_BASELINE_2_NOPD.txt', index = False)

forecasted_fugitive_ghg_rand_cpd = []
beta_rand = fugitive_rand_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['GDP_per_capita_3']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**3 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_rand_cpd.append(temp)

fugitive_forcdic_rand_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_rand_cpd}
fugitive_forc_df_rand_cpd = pd.DataFrame(fugitive_forcdic_rand_cpd)
print(fugitive_forc_df_rand_cpd)
fugitive_forc_df_rand_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_RANDOM_NOPD.txt', index = False)

forecasted_fugitive_ghg_rand2_cpd = []
beta_rand2 = fugitive_rand_results2.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    for i in range(100):
        try:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0] + beta_rand[state]))
        except:
            temp.append(max(0,beta_rand['Intercept'] + beta_rand['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i+22] + beta_rand['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i+22])**2 + beta_rand['Population_Density']*PD_df['forecasted_PD'][idx][0]))
    forecasted_fugitive_ghg_rand2_cpd.append(temp)

fugitive_forcdic_rand2_cpd = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg_rand2_cpd}
fugitive_forc_df_rand2_cpd = pd.DataFrame(fugitive_forcdic_rand2_cpd)
print(fugitive_forc_df_rand2_cpd)
fugitive_forc_df_rand2_cpd.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/fugitive_GHG_FORECAST_RANDOM_2_NOPD.txt', index = False)

# Aggregating fugitive production derived GHG emissions for the US

US_fugitive_GHG = np.zeros(22)
US_fugitive_GHG_fixed = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixed2 = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixedyr = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixedyr2 = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_init = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_init2 = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_rand = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_rand2 = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixed_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixed2_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixedyr_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_fixedyr2_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_init_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_init2_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_rand_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
US_fugitive_GHG_rand2_nopd = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
agg_fugitive = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Fugitive[i])
    agg_fugitive.append(temp)
    
for i in range(22):
    for j in range(len(agg_fugitive)):
        US_fugitive_GHG[i] += agg_fugitive[j][i]

for i in range(len(fugitive_forc_df)):
    for j in range(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0])):
        US_fugitive_GHG_fixed[j] += fugitive_forc_df['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixed2[j] += fugitive_forc_df2['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixedyr[j] += fugitive_forc_df_yr['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixedyr2[j] += fugitive_forc_df_yr2['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_init[j] += fugitive_forc_df_init['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_init2[j] += fugitive_forc_df_init2['Forecasted_fugitive_GHG'][i][j]        
        US_fugitive_GHG_rand[j] += fugitive_forc_df_rand['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_rand2[j] += fugitive_forc_df_rand2['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixed_nopd[j] += fugitive_forc_df_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixed2_nopd[j] += fugitive_forc_df2_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixedyr_nopd[j] += fugitive_forc_df_yr_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_fixedyr2_nopd[j] += fugitive_forc_df_yr2_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_init_nopd[j] += fugitive_forc_df_init_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_init2_nopd[j] += fugitive_forc_df_init2_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_rand_nopd[j] += fugitive_forc_df_rand_cpd['Forecasted_fugitive_GHG'][i][j]
        US_fugitive_GHG_rand2_nopd[j] += fugitive_forc_df_rand2_cpd['Forecasted_fugitive_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

models = [fugitive_fixed_results, fugitive_fixed_results2, fugitive_year_results, fugitive_year_results2, fugitive_results, fugitive_results2, fugitive_rand_results, fugitive_rand_results2]
residual_means = []
residual_std = []
for model in models:
    residual_means.append(np.mean(model.resid))
    residual_std.append(np.std(model.resid))
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for j in range(len(US_fugitive_GHG_init)):
    US_fugitive_GHG_fixed[i] = tobit_transform(US_fugitive_GHG_fixed[i], residual_std[0])
    US_fugitive_GHG_fixed2[i] = tobit_transform(US_fugitive_GHG_fixed2[i], residual_std[1])
    US_fugitive_GHG_fixedyr[i] = tobit_transform(US_fugitive_GHG_fixedyr[i], residual_std[2])
    US_fugitive_GHG_fixedyr2[i] = tobit_transform(US_fugitive_GHG_fixedyr2[i], residual_std[3])
    US_fugitive_GHG_init[i] = tobit_transform(US_fugitive_GHG_init[i], residual_std[4])
    US_fugitive_GHG_init2[i] = tobit_transform(US_fugitive_GHG_init2[i], residual_std[5])        
    US_fugitive_GHG_rand[i] = tobit_transform(US_fugitive_GHG_rand[i], residual_std[6])
    US_fugitive_GHG_rand2[i] = tobit_transform(US_fugitive_GHG_rand2[i], residual_std[7])        
    US_fugitive_GHG_fixed_nopd[i] = tobit_transform(US_fugitive_GHG_fixed_nopd[i], residual_std[0])
    US_fugitive_GHG_fixed2_nopd[i] = tobit_transform(US_fugitive_GHG_fixed2_nopd[i], residual_std[1])
    US_fugitive_GHG_fixedyr_nopd[i] = tobit_transform(US_fugitive_GHG_fixedyr_nopd[i], residual_std[2])
    US_fugitive_GHG_fixedyr2_nopd[i] = tobit_transform(US_fugitive_GHG_fixedyr2_nopd[i], residual_std[3])
    US_fugitive_GHG_init_nopd[i] = tobit_transform(US_fugitive_GHG_init_nopd[i], residual_std[4])
    US_fugitive_GHG_init2_nopd[i] = tobit_transform(US_fugitive_GHG_init2_nopd[i], residual_std[5])        
    US_fugitive_GHG_rand_nopd[i] = tobit_transform(US_fugitive_GHG_rand_nopd[i], residual_std[6])
    US_fugitive_GHG_rand2_nopd[i] = tobit_transform(US_fugitive_GHG_rand2_nopd[i], residual_std[7])

# Plotting subsector forecasts against estimated forecasts using percentages of full energy sector

# Because of the nature of these plots, this section is abandoned -- we will not use % estiamtes for anything, just use the subsector forecasts

# Plotting subsector forecasts for all subsectors grouped by model type along with corresponding full energy forecast from same model

plt.figure(36)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixed, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixed, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixed, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixed, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixed, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixed, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Fixed Effects Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixed.eps')

plt.figure(37)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed2, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixed2, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixed2, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixed2, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixed2, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixed2, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixed2, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Fixed Effects Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixed2.eps')

plt.figure(38)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixedyr, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixedyr, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixedyr, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixedyr, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixedyr, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixedyr, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Year Only Fixed Effects Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixedyr.eps')

plt.figure(39)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixedyr2, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr2, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixedyr2, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixedyr2, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixedyr2, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixedyr2, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixedyr2, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Year Only Fixed Effects Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixedyr2.eps')

plt.figure(40)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_init, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_init, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_init, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_init, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_init, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_init, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_init, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Baseline Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_init.eps')

plt.figure(41)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_init2, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_init2, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_init2, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_init2, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_init2, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_init2, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_init2, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Baseline Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_init2.eps')

plt.figure(42)
plt.ylim(bottom = 0)
plt.ylim(top = 9000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_rand, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_rand, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_rand, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_rand, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_rand, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_rand, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_rand, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Random Effects Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_rand.eps')

plt.figure(43)
plt.ylim(bottom = 0)
plt.ylim(top = 12000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_rand2, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_rand2, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_rand2, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_rand2, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_rand2, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_rand2, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_rand2, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Random Effects Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_rand2.eps')

plt.figure(44)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixed_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixed_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixed_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixed_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixed_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixed_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Fixed Effects Model without forecased PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixed_nopd.eps')

plt.figure(45)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed2_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixed2_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixed2_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixed2_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixed2_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixed2_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixed2_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Fixed Effects Model without cubic term or forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixed2_nopd.eps')

plt.figure(46)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixedyr_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixedyr_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixedyr_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixedyr_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixedyr_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixedyr_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Year Only Fixed Effects Model without forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixedyr_nopd.eps')

plt.figure(47)
plt.ylim(bottom = 0)
plt.ylim(top = 10000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixedyr2_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr2_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixedyr2_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_fixedyr2_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_fixedyr2_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixedyr2_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_fixedyr2_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Year Only Fixed Effects Model without cubic term or forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fixedyr2_nopd.eps')

plt.figure(48)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_init_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_init_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_init_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_init_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_init_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_init_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_init_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Baseline Model without forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_init_nopd.eps')

plt.figure(49)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_init2_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_init2_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_init2_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_init2_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_init2_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_init2_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_init2_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Baseline Model without cubic term or forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_init2_nopd.eps')

plt.figure(50)
plt.ylim(bottom = 0)
plt.ylim(top = 8500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_rand_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_rand_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_rand_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_rand_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_rand_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_rand_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_rand_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Random Effects Model without forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_rand_nopd.eps')

plt.figure(51)
plt.ylim(bottom = 0)
plt.ylim(top = 11500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_rand2_nopd, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG_rand2_nopd, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG_rand2_nopd, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG_rand2_nopd, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG_rand2_nopd, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG_rand2_nopd, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG_rand2_nopd, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions\n(Random Effects Model without cubic term or forecasted PD)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_rand2_nopd.eps')

# Plotting subsector forecasts for fixed subsector and all type

plt.figure(52)
plt.ylim(bottom = 0)
plt.ylim(top = 800)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_commercial_GHG_fixed2, label = 'Fixed Effects Model 2', color = cm(15))
plt.plot(basis, US_commercial_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(30))
plt.plot(basis, US_commercial_GHG_fixedyr2, label = 'Year Only Fixed Effects 2', color = cm(45))
plt.plot(basis, US_commercial_GHG_init, label = 'Baseline Model', color = cm(60))
plt.plot(basis, US_commercial_GHG_init2, label = 'Baseline Model 2', color = cm(75))
plt.plot(basis, US_commercial_GHG_rand, label = 'Random Effects Model', color = cm(90))
plt.plot(basis, US_commercial_GHG_rand2, label = 'Random Effects Model 2', color = cm(105))
plt.plot(basis, US_commercial_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(120))
plt.plot(basis, US_commercial_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(135))
plt.plot(basis, US_commercial_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(150))
plt.plot(basis, US_commercial_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(165))
plt.plot(basis, US_commercial_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(180))
plt.plot(basis, US_commercial_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(195))
plt.plot(basis, US_commercial_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(210))
plt.plot(basis, US_commercial_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(225))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Commercial Subsector\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_commercial_all.eps')

plt.figure(53)
plt.ylim(bottom = 0)
plt.ylim(top = 600)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_commercial_GHG_fixed2, label = 'Fixed Effects Model w/o cubic', color = cm(30))
plt.plot(basis, US_commercial_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(60))
plt.plot(basis, US_commercial_GHG_fixedyr2, label = 'Year Only Fixed Effects w/o cubic', color = cm(90))
plt.plot(basis, US_commercial_GHG_init, label = 'Baseline Model', color = cm(120))
plt.plot(basis, US_commercial_GHG_init2, label = 'Baseline Model w/o cubic', color = cm(150))
plt.plot(basis, US_commercial_GHG_rand, label = 'Random Effects Model', color = cm(180))
plt.plot(basis, US_commercial_GHG_rand2, label = 'Random Effects Model w/o cubic', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Commercial Subsector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_commercial_pd.eps')

plt.figure(54)
plt.ylim(bottom = 0)
plt.ylim(top = 800)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_commercial_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(0))
plt.plot(basis, US_commercial_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(30))
plt.plot(basis, US_commercial_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(60))
plt.plot(basis, US_commercial_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(90))
plt.plot(basis, US_commercial_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(120))
plt.plot(basis, US_commercial_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(150))
plt.plot(basis, US_commercial_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(180))
plt.plot(basis, US_commercial_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Commercial Subsector\n(Models without forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_commercial_nopd.eps')

plt.figure(55)
plt.ylim(bottom = 0)
plt.ylim(top = 1200)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_residential_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_residential_GHG_fixed2, label = 'Fixed Effects Model 2', color = cm(15))
plt.plot(basis, US_residential_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr2, label = 'Year Only Fixed Effects 2', color = cm(45))
plt.plot(basis, US_residential_GHG_init, label = 'Baseline Model', color = cm(60))
plt.plot(basis, US_residential_GHG_init2, label = 'Baseline Model 2', color = cm(75))
plt.plot(basis, US_residential_GHG_rand, label = 'Random Effects Model', color = cm(90))
plt.plot(basis, US_residential_GHG_rand2, label = 'Random Effects Model 2', color = cm(105))
plt.plot(basis, US_residential_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(120))
plt.plot(basis, US_residential_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(135))
plt.plot(basis, US_residential_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(150))
plt.plot(basis, US_residential_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(165))
plt.plot(basis, US_residential_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(180))
plt.plot(basis, US_residential_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(195))
plt.plot(basis, US_residential_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(210))
plt.plot(basis, US_residential_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(225))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Residential Subsector\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_residential_all.eps')

plt.figure(56)
plt.ylim(bottom = 0)
plt.ylim(top = 1000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_residential_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_residential_GHG_fixed2, label = 'Fixed Effects Model w/o cubic', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(60))
plt.plot(basis, US_residential_GHG_fixedyr2, label = 'Year Only Fixed Effects w/o cubic', color = cm(90))
plt.plot(basis, US_residential_GHG_init, label = 'Baseline Model', color = cm(120))
plt.plot(basis, US_residential_GHG_init2, label = 'Baseline Model w/o cubic', color = cm(150))
plt.plot(basis, US_residential_GHG_rand, label = 'Random Effects Model', color = cm(180))
plt.plot(basis, US_residential_GHG_rand2, label = 'Random Effects Model w/o cubic', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Residential Subsector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_residential_pd.eps')

plt.figure(57)
plt.ylim(bottom = 0)
plt.ylim(top = 1000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_residential_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(0))
plt.plot(basis, US_residential_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(30))
plt.plot(basis, US_residential_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(60))
plt.plot(basis, US_residential_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(90))
plt.plot(basis, US_residential_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(120))
plt.plot(basis, US_residential_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(150))
plt.plot(basis, US_residential_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(180))
plt.plot(basis, US_residential_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Residential Subsector\n(Models without forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_residential_nopd.eps')

plt.figure(58)
plt.ylim(bottom = 0)
plt.ylim(top = 3500)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_industrial_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_industrial_GHG_fixed2, label = 'Fixed Effects Model 2', color = cm(15))
plt.plot(basis, US_industrial_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(30))
plt.plot(basis, US_industrial_GHG_fixedyr2, label = 'Year Only Fixed Effects 2', color = cm(45))
plt.plot(basis, US_industrial_GHG_init, label = 'Baseline Model', color = cm(60))
plt.plot(basis, US_industrial_GHG_init2, label = 'Baseline Model 2', color = cm(75))
plt.plot(basis, US_industrial_GHG_rand, label = 'Random Effects Model', color = cm(90))
plt.plot(basis, US_industrial_GHG_rand2, label = 'Random Effects Model 2', color = cm(105))
plt.plot(basis, US_industrial_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(120))
plt.plot(basis, US_industrial_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(135))
plt.plot(basis, US_industrial_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(150))
plt.plot(basis, US_industrial_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(165))
plt.plot(basis, US_industrial_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(180))
plt.plot(basis, US_industrial_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(195))
plt.plot(basis, US_industrial_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(210))
plt.plot(basis, US_industrial_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(225))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Industrial Subsector\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_industrial_all.eps')

plt.figure(59)
plt.ylim(bottom = 0)
plt.ylim(top = 2700)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_industrial_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_industrial_GHG_fixed2, label = 'Fixed Effects Model w/o cubic', color = cm(30))
plt.plot(basis, US_industrial_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixedyr2, label = 'Year Only Fixed Effects w/o cubic', color = cm(90))
plt.plot(basis, US_industrial_GHG_init, label = 'Baseline Model', color = cm(120))
plt.plot(basis, US_industrial_GHG_init2, label = 'Baseline Model w/o cubic', color = cm(150))
plt.plot(basis, US_industrial_GHG_rand, label = 'Random Effects Model', color = cm(180))
plt.plot(basis, US_industrial_GHG_rand2, label = 'Random Effects Model w/o cubic', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Industrial Subsector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_industrial_pd.eps')

plt.figure(60)
plt.ylim(bottom = 0)
plt.ylim(top = 3200)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_industrial_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(0))
plt.plot(basis, US_industrial_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(30))
plt.plot(basis, US_industrial_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(60))
plt.plot(basis, US_industrial_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(90))
plt.plot(basis, US_industrial_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(120))
plt.plot(basis, US_industrial_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(150))
plt.plot(basis, US_industrial_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(180))
plt.plot(basis, US_industrial_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Industrial Subsector\n(Models without forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_industrial_nopd.eps')

plt.figure(61)
plt.ylim(bottom = 1000)
plt.ylim(top = 11000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_transportation_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_transportation_GHG_fixed2, label = 'Fixed Effects Model 2', color = cm(15))
plt.plot(basis, US_transportation_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(30))
plt.plot(basis, US_transportation_GHG_fixedyr2, label = 'Year Only Fixed Effects 2', color = cm(45))
plt.plot(basis, US_transportation_GHG_init, label = 'Baseline Model', color = cm(60))
plt.plot(basis, US_transportation_GHG_init2, label = 'Baseline Model 2', color = cm(75))
plt.plot(basis, US_transportation_GHG_rand, label = 'Random Effects Model', color = cm(90))
plt.plot(basis, US_transportation_GHG_rand2, label = 'Random Effects Model 2', color = cm(105))
plt.plot(basis, US_transportation_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(120))
plt.plot(basis, US_transportation_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(135))
plt.plot(basis, US_transportation_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(150))
plt.plot(basis, US_transportation_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(165))
plt.plot(basis, US_transportation_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(180))
plt.plot(basis, US_transportation_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(195))
plt.plot(basis, US_transportation_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(210))
plt.plot(basis, US_transportation_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(225))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Transportation Subsector\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_transportation_all.eps')

plt.figure(62)
plt.ylim(bottom = 1000)
plt.ylim(top = 9000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_transportation_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_transportation_GHG_fixed2, label = 'Fixed Effects Model w/o cubic', color = cm(30))
plt.plot(basis, US_transportation_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(60))
plt.plot(basis, US_transportation_GHG_fixedyr2, label = 'Year Only Fixed Effects w/o cubic', color = cm(90))
plt.plot(basis, US_transportation_GHG_init, label = 'Baseline Model', color = cm(120))
plt.plot(basis, US_transportation_GHG_init2, label = 'Baseline Model w/o cubic', color = cm(150))
plt.plot(basis, US_transportation_GHG_rand, label = 'Random Effects Model', color = cm(180))
plt.plot(basis, US_transportation_GHG_rand2, label = 'Random Effects Model w/o cubic', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Transportation Subsector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_transportation_pd.eps')

plt.figure(63)
plt.ylim(bottom = 1000)
plt.ylim(top = 9000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_transportation_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(0))
plt.plot(basis, US_transportation_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(30))
plt.plot(basis, US_transportation_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(60))
plt.plot(basis, US_transportation_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(90))
plt.plot(basis, US_transportation_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(120))
plt.plot(basis, US_transportation_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(150))
plt.plot(basis, US_transportation_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(180))
plt.plot(basis, US_transportation_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Transportation Subsector\n(Models without forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_transportation_nopd.eps')

plt.figure(64)
plt.ylim(bottom = -500)
plt.ylim(top = 3000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_electric_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_electric_GHG_fixed2, label = 'Fixed Effects Model 2', color = cm(15))
plt.plot(basis, US_electric_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(30))
plt.plot(basis, US_electric_GHG_fixedyr2, label = 'Year Only Fixed Effects 2', color = cm(45))
plt.plot(basis, US_electric_GHG_init, label = 'Baseline Model', color = cm(60))
plt.plot(basis, US_electric_GHG_init2, label = 'Baseline Model 2', color = cm(75))
plt.plot(basis, US_electric_GHG_rand, label = 'Random Effects Model', color = cm(90))
plt.plot(basis, US_electric_GHG_rand2, label = 'Random Effects Model 2', color = cm(105))
plt.plot(basis, US_electric_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(120))
plt.plot(basis, US_electric_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(135))
plt.plot(basis, US_electric_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(150))
plt.plot(basis, US_electric_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(165))
plt.plot(basis, US_electric_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(180))
plt.plot(basis, US_electric_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(195))
plt.plot(basis, US_electric_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(210))
plt.plot(basis, US_electric_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(225))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Electric Power Subsector\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_electric_all.eps')

plt.figure(65)
plt.ylim(bottom = 0)
plt.ylim(top = 3000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_electric_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_electric_GHG_fixed2, label = 'Fixed Effects Model w/o cubic', color = cm(30))
plt.plot(basis, US_electric_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(60))
plt.plot(basis, US_electric_GHG_fixedyr2, label = 'Year Only Fixed Effects w/o cubic', color = cm(90))
plt.plot(basis, US_electric_GHG_init, label = 'Baseline Model', color = cm(120))
plt.plot(basis, US_electric_GHG_init2, label = 'Baseline Model w/o cubic', color = cm(150))
plt.plot(basis, US_electric_GHG_rand, label = 'Random Effects Model', color = cm(180))
plt.plot(basis, US_electric_GHG_rand2, label = 'Random Effects Model w/o cubic', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Electric Power Subsector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_electric_pd.eps')

plt.figure(66)
plt.ylim(bottom = 0)
plt.ylim(top = 3000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_electric_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(0))
plt.plot(basis, US_electric_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(30))
plt.plot(basis, US_electric_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(60))
plt.plot(basis, US_electric_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(90))
plt.plot(basis, US_electric_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(120))
plt.plot(basis, US_electric_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(150))
plt.plot(basis, US_electric_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(180))
plt.plot(basis, US_electric_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from the Electric Power Subsector\n(Models without forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_electric_nopd.eps')

plt.figure(67)
plt.ylim(bottom = 0)
plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_fugitive_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_fugitive_GHG_fixed2, label = 'Fixed Effects Model 2', color = cm(15))
plt.plot(basis, US_fugitive_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(30))
plt.plot(basis, US_fugitive_GHG_fixedyr2, label = 'Year Only Fixed Effects 2', color = cm(45))
plt.plot(basis, US_fugitive_GHG_init, label = 'Baseline Model', color = cm(60))
plt.plot(basis, US_fugitive_GHG_init2, label = 'Baseline Model 2', color = cm(75))
plt.plot(basis, US_fugitive_GHG_rand, label = 'Random Effects Model', color = cm(90))
plt.plot(basis, US_fugitive_GHG_rand2, label = 'Random Effects Model 2', color = cm(105))
plt.plot(basis, US_fugitive_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(120))
plt.plot(basis, US_fugitive_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(135))
plt.plot(basis, US_fugitive_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(150))
plt.plot(basis, US_fugitive_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(165))
plt.plot(basis, US_fugitive_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(180))
plt.plot(basis, US_fugitive_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(195))
plt.plot(basis, US_fugitive_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(210))
plt.plot(basis, US_fugitive_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(225))

# Add titles and save
plt.title('Subsector Level GHG Emissions from Fugitive Emissions\n(All Models)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fugitive_all.eps')

plt.figure(68)
plt.ylim(bottom = 0)
plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_fugitive_GHG_fixed, label = 'Fixed Effects Model', color = cm(0))
plt.plot(basis, US_fugitive_GHG_fixed2, label = 'Fixed Effects Model w/o cubic', color = cm(30))
plt.plot(basis, US_fugitive_GHG_fixedyr, label = 'Year Only Fixed Effects', color = cm(60))
plt.plot(basis, US_fugitive_GHG_fixedyr2, label = 'Year Only Fixed Effects w/o cubic', color = cm(90))
plt.plot(basis, US_fugitive_GHG_init, label = 'Baseline Model', color = cm(120))
plt.plot(basis, US_fugitive_GHG_init2, label = 'Baseline Model w.o cubic', color = cm(150))
plt.plot(basis, US_fugitive_GHG_rand, label = 'Random Effects Model', color = cm(180))
plt.plot(basis, US_fugitive_GHG_rand2, label = 'Random Effects Model w/o cubic', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from Fugitive Emissions', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fugitive_pd.eps')

plt.figure(69)
plt.ylim(bottom = 0)
plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_fugitive_GHG_fixed_nopd, label = 'Fixed Effects Model - no PD', color = cm(0))
plt.plot(basis, US_fugitive_GHG_fixed2_nopd, label = 'Fixed Effects Model 2 - no PD', color = cm(30))
plt.plot(basis, US_fugitive_GHG_fixedyr_nopd, label = 'Year Only Fixed Effects - no PD', color = cm(60))
plt.plot(basis, US_fugitive_GHG_fixedyr2_nopd, label = 'Year Only Fixed Effects 2 - no PD', color = cm(90))
plt.plot(basis, US_fugitive_GHG_init_nopd, label = 'Baseline Model - no PD', color = cm(120))
plt.plot(basis, US_fugitive_GHG_init2_nopd, label = 'Baseline Model 2 - no PD', color = cm(150))
plt.plot(basis, US_fugitive_GHG_rand_nopd, label = 'Random Effects Model - no PD', color = cm(180))
plt.plot(basis, US_fugitive_GHG_rand2_nopd, label = 'Random Effects Model 2 - no PD', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions from Fugitive Emissions\n(Models without forecasted population density)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/subsector_plots_fugitive_nopd.eps')

# Lastly, aggregate subsector emissions forecasts and compare to energy emissions forecasts

subsector_aggregated_fixed = US_commercial_GHG_fixed + US_residential_GHG_fixed + US_industrial_GHG_fixed + US_transportation_GHG_fixed + US_electric_GHG_fixed + US_fugitive_GHG_fixed
subsector_aggregated_fixed2 = US_commercial_GHG_fixed2 + US_residential_GHG_fixed2 + US_industrial_GHG_fixed2 + US_transportation_GHG_fixed2 + US_electric_GHG_fixed2 + US_fugitive_GHG_fixed2
subsector_aggregated_fixedyr = US_commercial_GHG_fixedyr + US_residential_GHG_fixedyr + US_industrial_GHG_fixedyr + US_transportation_GHG_fixedyr + US_electric_GHG_fixedyr + US_fugitive_GHG_fixedyr
subsector_aggregated_fixedyr2 = US_commercial_GHG_fixedyr2 + US_residential_GHG_fixedyr2 + US_industrial_GHG_fixedyr2 + US_transportation_GHG_fixedyr2 + US_electric_GHG_fixedyr2 + US_fugitive_GHG_fixedyr2
subsector_aggregated_init = US_commercial_GHG_init + US_residential_GHG_init + US_industrial_GHG_init + US_transportation_GHG_init + US_electric_GHG_init + US_fugitive_GHG_init
subsector_aggregated_init2 = US_commercial_GHG_init2 + US_residential_GHG_init2 + US_industrial_GHG_init2 + US_transportation_GHG_init2 + US_electric_GHG_init2 + US_fugitive_GHG_init2
subsector_aggregated_rand = US_commercial_GHG_rand + US_residential_GHG_rand + US_industrial_GHG_rand + US_transportation_GHG_rand + US_electric_GHG_rand + US_fugitive_GHG_rand
subsector_aggregated_rand2 = US_commercial_GHG_rand2 + US_residential_GHG_rand2 + US_industrial_GHG_rand2 + US_transportation_GHG_rand2 + US_electric_GHG_rand2 + US_fugitive_GHG_rand2

plt.figure(70)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_fixed, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_fixed, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Fixed Effects Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_fixed.eps')

plt.figure(71)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_fixed2, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_fixed2, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Fixed Effects Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_fixed2.eps')

plt.figure(72)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_fixedyr, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_fixedyr, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Year Only Fixed Effects Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_fixedyr.eps')

plt.figure(73)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_fixedyr2, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_fixedyr2, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Year Only Fixed Effects Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_fixedyr2.eps')

plt.figure(74)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_init, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Baseline Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_init.eps')

plt.figure(75)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init2, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_init2, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Baseline Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_init2.eps')

plt.figure(76)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_rand, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_rand, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Random Effects Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_rand.eps')

plt.figure(77)
#plt.ylim(bottom = 0)
#plt.ylim(top = 350)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_rand2, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated_rand2, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Random Effects Model without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_rand2.eps')

plt.figure(78)
plt.ylim(bottom = 4000)
plt.ylim(top = 8000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_init, label = 'Baseline', color = cm(30))
plt.plot(basis, subsector_aggregated_init, label = 'Aggregated Baseline', color = cm(165))
plt.plot(basis, US_energy_GHG_init2, label = 'Baseline w/o cubic', color = cm(30), marker = '.')
plt.plot(basis, subsector_aggregated_init2, label = 'Aggregated Baseline w/o cubic', color = cm(165), marker = '.')
plt.plot(basis, US_energy_GHG_fixedyr, label = 'Year Fixed Effects', color = cm(195))
plt.plot(basis, subsector_aggregated_fixedyr, label = 'Aggregated Year Fixed Effects w/o cubic', color = cm(0))
plt.plot(basis, US_energy_GHG_fixedyr2, label = 'Year Fixed Effects w/o cubic', color = cm(0), marker = '.')
plt.plot(basis, subsector_aggregated_fixedyr2, label = 'Aggregated Year Fixed Effets w/o cubic', color = cm(195), marker = '.')

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Comparing forecasts with and without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 3, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_cubic_comp_pt1.eps')

plt.figure(79)
plt.ylim(bottom = 5000)
plt.ylim(top = 11000)
basis = [i for i in range(2012,2112)]
plt.plot(basis, US_energy_GHG_fixed, label = 'Fixed Effects', color = cm(30))
plt.plot(basis, subsector_aggregated_fixed, label = 'Aggregated Fixed Effects', color = cm(165))
plt.plot(basis, US_energy_GHG_fixed2, label = 'Fixed Effects w/o cubic', color = cm(30), marker = '.')
plt.plot(basis, subsector_aggregated_fixed2, label = 'Aggregated Fixed Effects w/o cubic', color = cm(165), marker = '.')
plt.plot(basis, US_energy_GHG_rand, label = 'Random Effects', color = cm(0))
plt.plot(basis, subsector_aggregated_rand, label = 'Aggregated Random Effects', color = cm(195))
plt.plot(basis, US_energy_GHG_rand2, label = 'Random Effects w/o cubic', color = cm(0), marker = '.')
plt.plot(basis, subsector_aggregated_rand2, label = 'Aggregated Random Effects w/o cubic', color = cm(195), marker = '.')

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(Comparing forecasts with and without cubic term)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 0, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/agg_subsector_v_sector_cubic_comp_pt2.eps')

# Testing the various models for aggregated in-sample data fitting

# Calculate the data mean

Y_bar = [sum(Y_Energy)/len(Y_Energy)]*len(Y_Energy)

# Calculate aggregate predicted values

agg_val_init = electric_results.fittedvalues + transportation_results.fittedvalues + industrial_results.fittedvalues + commercial_results.fittedvalues + residential_results.fittedvalues + fugitive_results.fittedvalues
agg_val_init2 = electric_results2.fittedvalues + transportation_results2.fittedvalues + industrial_results2.fittedvalues + commercial_results2.fittedvalues + residential_results2.fittedvalues + fugitive_results2.fittedvalues
agg_val_year = electric_year_results.fittedvalues + transportation_year_results.fittedvalues + industrial_year_results.fittedvalues + commercial_year_results.fittedvalues + residential_year_results.fittedvalues + fugitive_year_results.fittedvalues
agg_val_year2 = electric_year_results2.fittedvalues + transportation_year_results2.fittedvalues + industrial_year_results2.fittedvalues + commercial_year_results2.fittedvalues + residential_year_results2.fittedvalues + fugitive_year_results2.fittedvalues
agg_val_fixed = electric_fixed_results.fittedvalues + transportation_fixed_results.fittedvalues + industrial_fixed_results.fittedvalues + commercial_fixed_results.fittedvalues + residential_fixed_results.fittedvalues + fugitive_fixed_results.fittedvalues
agg_val_fixed2 = electric_fixed_results2.fittedvalues + transportation_fixed_results2.fittedvalues + industrial_fixed_results2.fittedvalues + commercial_fixed_results2.fittedvalues + residential_fixed_results2.fittedvalues + fugitive_fixed_results2.fittedvalues

# Calculate SSR

agg_sr_init = (agg_val_init - Y_bar)**2
agg_sr_init2 = (agg_val_init2 - Y_bar)**2
agg_sr_year = (agg_val_year - Y_bar)**2
agg_sr_year2 = (agg_val_year2 - Y_bar)**2
agg_sr_fixed = (agg_val_fixed - Y_bar)**2
agg_sr_fixed2 = (agg_val_fixed2 - Y_bar)**2

agg_ssr_init = sum(agg_sr_init)
agg_ssr_init2 = sum(agg_sr_init2)
agg_ssr_year = sum(agg_sr_year)
agg_ssr_year2 = sum(agg_sr_year2)
agg_ssr_fixed = sum(agg_sr_fixed)
agg_ssr_fixed2 = sum(agg_sr_fixed2)

# Calculate SSTO

agg_sto_init = (Y_Energy - Y_bar)**2
agg_sto_init2 = (Y_Energy - Y_bar)**2
agg_sto_year = (Y_Energy - Y_bar)**2
agg_sto_year2 = (Y_Energy - Y_bar)**2
agg_sto_fixed = (Y_Energy - Y_bar)**2
agg_sto_fixed2 = (Y_Energy - Y_bar)**2

agg_ssto_init = sum(agg_sto_init)
agg_ssto_init2 = sum(agg_sto_init2)
agg_ssto_year = sum(agg_sto_year)
agg_ssto_year2 = sum(agg_sto_year2)
agg_ssto_fixed = sum(agg_sto_fixed)
agg_ssto_fixed2 = sum(agg_sto_fixed2)

# Calculate $R^{2}$

agg_r2_init = agg_ssr_init / agg_ssto_init
agg_r2_init2 = agg_ssr_init2 / agg_ssto_init2
agg_r2_year = agg_ssr_year / agg_ssto_year
agg_r2_year2 = agg_ssr_year2 / agg_ssto_year2
agg_r2_fixed = agg_ssr_fixed / agg_ssto_fixed
agg_r2_fixed2 = agg_ssr_fixed2 / agg_ssto_fixed2

# Calculate Adjusted $R^{2}$

agg_ar2_init = 1 - ((1 - agg_r2_init) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_results.df_model + 1))))
agg_ar2_init2 = 1 - ((1 - agg_r2_init2) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_results2.df_model + 1))))
agg_ar2_year = 1 - ((1 - agg_r2_year) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_year_results.df_model + 1))))
agg_ar2_year2 = 1 - ((1 - agg_r2_year2) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_year_results2.df_model + 1))))
agg_ar2_fixed = 1 - ((1 - agg_r2_fixed) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_fixed_results.df_model + 1))))
agg_ar2_fixed2 = 1 - ((1 - agg_r2_fixed2) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_fixed_results2.df_model + 1))))

# Get Adjusted $R^{2}$ from independent energy sector models

ind_ar2_init = energy_results.rsquared_adj
ind_ar2_init2 = energy_results2.rsquared_adj
ind_ar2_year = energy_year_results.rsquared_adj
ind_ar2_year2 = energy_year_results2.rsquared_adj
ind_ar2_fixed = energy_fixed_results.rsquared_adj
ind_ar2_fixed2 =energy_fixed_results2.rsquared_adj 

# Create dataframe of results and write to file

models = pd.DataFrame(['Baseline', 'Baseline2', 'Year', 'Year2', 'Fixed', 'Fixed2'])
ind = pd.DataFrame([ind_ar2_init, ind_ar2_init2, ind_ar2_year, ind_ar2_year2, ind_ar2_fixed, ind_ar2_fixed2])
agg = pd.DataFrame([agg_ar2_init, agg_ar2_init2, agg_ar2_year, agg_ar2_year2, agg_ar2_fixed, agg_ar2_fixed2])
adj_r2_df = pd.concat([models, ind, agg], axis = 1)
adj_r2_df.columns = ['Model', 'Independent', 'Aggregate']
adj_r2_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/adjusted_rsquared.txt', index = False)




