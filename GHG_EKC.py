# This script uses an AR1 autoregressive model to forecast aggregated subsector level emissions of GHG in the US

# Importing required modules

import numpy as np
import pandas as pd
import statsmodels.api as stats
import matplotlib.pyplot as plt
from scipy.stats import norm

# Declaring filepaths for data files

arpath = 'C:/Users/User/Documents/Data/state_level_ardata.csv'
epapath = 'C:/Users/User/Documents/Data/EPAGHGdata.csv'
gdp_filepath = 'C:/Users/User/Documents/Data/gdp_reg_data.csv'
ratiopath = 'C:/Users/User/Documents/Data/ghg_ratio_data.csv'

# Structuring the dataframes for regression

data = pd.read_csv(arpath)
state_dummies = pd.get_dummies(data['State'])

Y_Energy = data['Energy_Delta']
Y_Commercial = data['Commercial_Delta']
Y_Residential = data['Residential_Delta']
Y_Industrial = data['Industrial_Delta']
Y_Transportation = data['Transportation_Delta']
Y_Electric_Power = data['Electric_Power_Delta']
Y_Fugitive = data['Fugitive_Delta']

X_Energy = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Commercial = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Residential = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Industrial = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Transportation = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Electric_Power = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]
X_Fugitive = data[['GDP_per_capita', 'GDP_per_capita_2', 'Population_Density', 'Renewables', 'HDD', 'CDD']]

# Running AR-1 regression models with EKC Hypothesis format

# Emissions from energy

energy_model = stats.OLS(Y_Energy.astype(float), X_Energy.astype(float))
energy_results = energy_model.fit()
print(energy_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/energy_model.txt', 'w')
file.write(energy_results.summary().as_text())
file.close()

# Emissions from commercial

commercial_model = stats.OLS(Y_Commercial.astype(float), X_Commercial.astype(float))
commercial_results = commercial_model.fit()
print(commercial_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/commercial_model.txt', 'w')
file.write(commercial_results.summary().as_text())
file.close()

# Emissions from residential

residential_model = stats.OLS(Y_Residential.astype(float), X_Residential.astype(float))
residential_results = residential_model.fit()
print(residential_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/residential_model.txt', 'w')
file.write(residential_results.summary().as_text())
file.close()

# Emissions from industry

industrial_model = stats.OLS(Y_Industrial.astype(float), X_Industrial.astype(float))
industrial_results = industrial_model.fit()
print(industrial_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/industrial_model.txt', 'w')
file.write(industrial_results.summary().as_text())
file.close()

# Emissions from transportation

transportation_model = stats.OLS(Y_Transportation.astype(float), X_Transportation.astype(float))
transportation_results = transportation_model.fit()
print(transportation_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/transportation_model.txt', 'w')
file.write(transportation_results.summary().as_text())
file.close()

# Emissions from electric power

electric_model = stats.OLS(Y_Electric_Power.astype(float), X_Electric_Power.astype(float))
electric_results = electric_model.fit()
print(electric_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/electric_model.txt', 'w')
file.write(electric_results.summary().as_text())
file.close()

# Fugitive emissions in energy

fugitive_model = stats.OLS(Y_Fugitive.astype(float), X_Fugitive.astype(float))
fugitive_results = fugitive_model.fit()
print(fugitive_results.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/fugitive_model.txt', 'w')
file.write(fugitive_results.summary().as_text())
file.close()

# Forecasting Population Density

# Generate list of states and obtain parameters for forecasts

States = list(data.State.unique())

# Find time trend for population density

pop_params = [[],[]]

y = [i for i in range(21)]
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

# Forecasting Fossil Fuel Usage

# Find time trend for fossil fuel usage

ff_params = [[],[]]

for state in States:
    ff_temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            ff_temp.append(data.Renewables[i])
    temp_ff_mod = stats.OLS(ff_temp, y)
    temp_ff_res = temp_ff_mod.fit()
    ff_params[0].append(temp_ff_res.params[0])
    ff_params[1].append(temp_ff_res.params[1])

# Create dataframe with pandas to record trends

dff = {'State':States, 'FF_int':ff_params[0], 'FF_slope':ff_params[1]}
ffparams_df = pd.DataFrame(dff)
print(ffparams_df)

# Forecasting Heating Degree Days

# Find time trend for HDD

HDD_params = [[],[]]

for state in States:
    HDD_temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            HDD_temp.append(data.HDD[i])
    temp_HDD_mod = stats.OLS(HDD_temp, y)
    temp_HDD_res = temp_HDD_mod.fit()
    HDD_params[0].append(temp_HDD_res.params[0])
    HDD_params[1].append(temp_HDD_res.params[1])

# Create dataframe with pandas to record trends

HDD = {'State':States, 'HDD_int':HDD_params[0], 'HDD_slope':HDD_params[1]}
HDDparams_df = pd.DataFrame(HDD)
print(HDDparams_df)

# Forecasting Cooling Degree Days

# Find time trend for CDD

CDD_params = [[],[]]

for state in States:
    CDD_temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            CDD_temp.append(data.CDD[i])
    temp_CDD_mod = stats.OLS(CDD_temp, y)
    temp_CDD_res = temp_CDD_mod.fit()
    CDD_params[0].append(temp_CDD_res.params[0])
    CDD_params[1].append(temp_CDD_res.params[1])

# Create dataframe with pandas to record trends

CDD = {'State':States, 'CDD_int':CDD_params[0], 'CDD_slope':CDD_params[1]}
CDDparams_df = pd.DataFrame(CDD)
print(CDDparams_df)

# Forecast state level total GHG emissions

# First estimate state level GDP growth rates via the following model

# \ln{\left(\frac{y_{i,t}}{y_{i,t-1}}\right)} = \gamma_{0,t} + \gamma_{1}\ln{(y_{i,t-1})} + \gamma_{2}\ln{(y_{i,t-1}^{2})} + \theta_{i,t}

# Create dataframe with all data needed for this regression

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
GDP_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/gdp_forecast.txt', index = False)

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
    for i in range(21,121):
        fgdp1.append((fgdp1[i-1] + gdp_data2['gamma_0'][idx] + gdp_data2['gamma_1'][idx]*fgdp1[i-1] + gdp_data2['gamma_2'][idx]*fgdp2[i-1]))
        fgdp.append(np.exp(fgdp1[i]))
        fgdp2.append(np.log(fgdp[i]**2))
    gdp_forc.append(fgdp)

gdpforcdic = {'State':States, 'forecasted_gdp_per_capita':gdp_forc}
GDP_df = pd.DataFrame(gdpforcdic)
GDP_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/gdp_forecast.txt', index = False)

# Second estimate future population densities, fossil fuel usages, HDD, and CDD

pop_den_forecast = []
ff_forecast = []
HDD_forecast = []
CDD_forecast = []

for state in States:
    idx = params_df[params_df['State'] == state].index.values.astype(int)[0]
    temp = [(params_df['PD_int'][idx] + params_df['PD_slope'][idx]*i) for i in range(22,122)]
    temp2 = [(ffparams_df['FF_int'][idx] + ffparams_df['FF_slope'][idx]*i) for i in range (22,122)]
    tempH = [(HDDparams_df['HDD_int'][idx] + HDDparams_df['HDD_slope'][idx]*i) for i in range (22,122)]
    tempC = [(CDDparams_df['CDD_int'][idx] + CDDparams_df['CDD_slope'][idx]*i) for i in range (22,122)]
    pop_den_forecast.append(temp)
    ff_forecast.append(temp2)
    HDD_forecast.append(tempH)
    CDD_forecast.append(tempC)

pdforc = {'State':States, 'forecasted_PD':pop_den_forecast, 'forecasted_FF':ff_forecast, 'forecasted_HDD':HDD_forecast, 'forecasted_CDD':CDD_forecast}
PD_df = pd.DataFrame(pdforc)
PD_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/pd_ff_forecast.txt', index = False)

# Third get raw estimates of state level forecasted emissions

forecasted_energy_ghg = []
ar_usage = data.iloc[960::].set_index('State')
beta = energy_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Energy[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_energy_ghg.append(temp)

energy_forcdic = {'State':States, 'Forecasted_Energy_GHG':forecasted_energy_ghg}
energy_forc_df = pd.DataFrame(energy_forcdic)
print(energy_forc_df)
energy_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/ENERGY_GHG_FORECAST.txt', index = False)

# Aggregating energy production derived GHG emissions for the US

historical = np.zeros(21)
US_energy_GHG = np.zeros(len(energy_forc_df['Forecasted_Energy_GHG'][0]))
agg_energy = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Energy[i])
    agg_energy.append(temp)
    
for i in range(21):
    for j in range(len(agg_energy)):
        historical[i] += agg_energy[j][i]

for i in range(len(energy_forc_df)):
    for j in range(len(energy_forc_df['Forecasted_Energy_GHG'][0])):
        US_energy_GHG[j] += energy_forc_df['Forecasted_Energy_GHG'][i][j]

ratiodata = pd.read_csv(ratiopath)
ratio_mean = np.mean(ratiodata.Ratio)
US_energy_GHG = US_energy_GHG / ratio_mean

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(energy_results.resid)
residual_std = np.std(energy_results.resid)
        
# Estimate the tobit transformed forecasts

# Defining function for the tobit transformation

def tobit_transform(x,s):
    out = x*norm.cdf(x/s) + s*norm.pdf(x)
    return out

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_energy_GHG)):
    US_energy_GHG[i] = tobit_transform(US_energy_GHG[i], residual_std)

# Create a scatter plot of historical aggregated US GHG emissions from energy production
    
from pylab import rcParams
rcParams['figure.figsize'] = 8.5, 8.5
cm = plt.get_cmap('gist_rainbow')

plt.figure(0)
basis = [i for i in range(1991,2012)]
plt.plot(basis, historical, label = 'Historical Data')

# Add titles
plt.title('Historical Data', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/historical.eps')

# Create scatter plots for forecasted aggreagated US GHG emissions from energy production

plt.figure(1)
basis = [i for i in range(2012,2018)]
plt.plot(basis, US_energy_GHG, label = 'AR-1 EKC', color = cm(00))

# Add legend
plt.legend(loc = 8, ncol = 2)

# Add titles
plt.title('Aggregated GHG Emissions from Energy Production', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')

# Save the figure

plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/forecasts_all.eps')

# State level plots

plt.figure(2)
for i in range(len(States)):
    plt.plot(basis, energy_forc_df['Forecasted_Energy_GHG'][i])

# Add titles and save
plt.title('State-wise GHG Emissions from Energy Production', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/all_states_model.eps')

# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_commercial_ghg = []
beta = commercial_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Commercial[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_commercial_ghg.append(temp)

commercial_forcdic = {'State':States, 'Forecasted_commercial_GHG':forecasted_commercial_ghg}
commercial_forc_df = pd.DataFrame(commercial_forcdic)
print(commercial_forc_df)
commercial_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/commercial_GHG_FORECAST.txt', index = False)

# Aggregating commercial production derived GHG emissions for the US

US_commercial_historical = np.zeros(21)
US_commercial_GHG = np.zeros(len(commercial_forc_df['Forecasted_commercial_GHG'][0]))
agg_commercial = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Commercial[i])
    agg_commercial.append(temp)
    
for i in range(21):
    for j in range(len(agg_commercial)):
        US_commercial_historical[i] += agg_commercial[j][i]

for i in range(len(commercial_forc_df)):
    for j in range(len(commercial_forc_df['Forecasted_commercial_GHG'][0])):
        US_commercial_GHG[j] += commercial_forc_df['Forecasted_commercial_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(commercial_results.resid)
residual_std = np.std(commercial_results.resid)

# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_commercial_GHG)):
    US_commercial_GHG[i] = tobit_transform(US_commercial_GHG[i], residual_std)
    
# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_residential_ghg = []
beta = residential_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Residential[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_residential_ghg.append(temp)

residential_forcdic = {'State':States, 'Forecasted_residential_GHG':forecasted_residential_ghg}
residential_forc_df = pd.DataFrame(residential_forcdic)
print(residential_forc_df)
residential_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/residential_GHG_FORECAST.txt', index = False)

# Aggregating residential production derived GHG emissions for the US

US_residential_historical = np.zeros(21)
US_residential_GHG = np.zeros(len(residential_forc_df['Forecasted_residential_GHG'][0]))
agg_residential = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Residential[i])
    agg_residential.append(temp)
    
for i in range(21):
    for j in range(len(agg_residential)):
        US_residential_historical[i] += agg_residential[j][i]

for i in range(len(residential_forc_df)):
    for j in range(len(residential_forc_df['Forecasted_residential_GHG'][0])):
        US_residential_GHG[j] += residential_forc_df['Forecasted_residential_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(residential_results.resid)
residual_std = np.std(residential_results.resid)
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_residential_GHG)):
    US_residential_GHG[i] = tobit_transform(US_residential_GHG[i], residual_std)
    
# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_industrial_ghg = []
beta = industrial_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Industrial[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_industrial_ghg.append(temp)

industrial_forcdic = {'State':States, 'Forecasted_industrial_GHG':forecasted_industrial_ghg}
industrial_forc_df = pd.DataFrame(industrial_forcdic)
print(industrial_forc_df)
industrial_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/industrial_GHG_FORECAST.txt', index = False)

# Aggregating industrial production derived GHG emissions for the US

US_industrial_historical = np.zeros(21)
US_industrial_GHG = np.zeros(len(industrial_forc_df['Forecasted_industrial_GHG'][0]))
agg_industrial = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Industrial[i])
    agg_industrial.append(temp)
    
for i in range(21):
    for j in range(len(agg_industrial)):
        US_industrial_historical[i] += agg_industrial[j][i]

for i in range(len(industrial_forc_df)):
    for j in range(len(industrial_forc_df['Forecasted_industrial_GHG'][0])):
        US_industrial_GHG[j] += industrial_forc_df['Forecasted_industrial_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(industrial_results.resid)
residual_std = np.std(industrial_results.resid)
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_industrial_GHG)):
    US_industrial_GHG[i] = tobit_transform(US_industrial_GHG[i], residual_std)
    
# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_transportation_ghg = []
beta = transportation_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Transportation[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_transportation_ghg.append(temp)

transportation_forcdic = {'State':States, 'Forecasted_transportation_GHG':forecasted_transportation_ghg}
transportation_forc_df = pd.DataFrame(transportation_forcdic)
print(transportation_forc_df)
transportation_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/transportation_GHG_FORECAST.txt', index = False)

# Aggregating transportation production derived GHG emissions for the US

US_transportation_historical = np.zeros(21)
US_transportation_GHG = np.zeros(len(transportation_forc_df['Forecasted_transportation_GHG'][0]))
agg_transportation = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Transportation[i])
    agg_transportation.append(temp)
    
for i in range(21):
    for j in range(len(agg_transportation)):
        US_transportation_historical[i] += agg_transportation[j][i]

for i in range(len(transportation_forc_df)):
    for j in range(len(transportation_forc_df['Forecasted_transportation_GHG'][0])):
        US_transportation_GHG[j] += transportation_forc_df['Forecasted_transportation_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(transportation_results.resid)
residual_std = np.std(transportation_results.resid)
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_transportation_GHG)):
    US_transportation_GHG[i] = tobit_transform(US_transportation_GHG[i], residual_std)
    
# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_electric_ghg = []
beta = electric_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Electric_Power[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_electric_ghg.append(temp)

electric_forcdic = {'State':States, 'Forecasted_electric_GHG':forecasted_electric_ghg}
electric_forc_df = pd.DataFrame(electric_forcdic)
print(electric_forc_df)
electric_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/electric_GHG_FORECAST.txt', index = False)

# Aggregating electric production derived GHG emissions for the US

US_electric_historical = np.zeros(21)
US_electric_GHG = np.zeros(len(electric_forc_df['Forecasted_electric_GHG'][0]))
agg_electric = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Electric_Power[i])
    agg_electric.append(temp)
    
for i in range(21):
    for j in range(len(agg_electric)):
        US_electric_historical[i] += agg_electric[j][i]

for i in range(len(electric_forc_df)):
    for j in range(len(electric_forc_df['Forecasted_electric_GHG'][0])):
        US_electric_GHG[j] += electric_forc_df['Forecasted_electric_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(electric_results.resid)
residual_std = np.std(electric_results.resid)

# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_electric_GHG)):
    US_electric_GHG[i] = tobit_transform(US_electric_GHG[i], residual_std)
    
# Do subsetor analyses to provide a more in depth analysis of energy production based GHG emissions

# Get raw estimates of state level forecasted emissions for the following cases:

forecasted_fugitive_ghg = []
beta = fugitive_results.params

for state in States:
    temp = []
    idx = GDP_df[GDP_df['State'] == state].index.values.astype(int)[0]
    lag_val = ar_usage.Fugitive[state]
    for i in range(6):
        try:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i] + beta[state]))
            lag_val = temp[len(temp)-1]
        except:
            temp.append(max(0, lag_val + beta['GDP_per_capita']*GDP_df['forecasted_gdp_per_capita'][idx][i] + beta['GDP_per_capita_2']*(GDP_df['forecasted_gdp_per_capita'][idx][i])**2 + beta['Population_Density']*PD_df['forecasted_PD'][idx][i] + beta['Renewables']*PD_df['forecasted_FF'][idx][i]))
            lag_val = temp[len(temp)-1]
    forecasted_fugitive_ghg.append(temp)

fugitive_forcdic = {'State':States, 'Forecasted_fugitive_GHG':forecasted_fugitive_ghg}
fugitive_forc_df = pd.DataFrame(fugitive_forcdic)
print(fugitive_forc_df)
fugitive_forc_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/fugitive_GHG_FORECAST.txt', index = False)

# Aggregating fugitive production derived GHG emissions for the US

US_fugitive_historical = np.zeros(21)
US_fugitive_GHG = np.zeros(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0]))
agg_fugitive = []

for state in States:
    temp = []
    for i in range(len(data)):
        if data.State[i] == state:
            temp.append(data.Fugitive[i])
    agg_fugitive.append(temp)
    
for i in range(21):
    for j in range(len(agg_fugitive)):
        US_fugitive_historical[i] += agg_fugitive[j][i]

for i in range(len(fugitive_forc_df)):
    for j in range(len(fugitive_forc_df['Forecasted_fugitive_GHG'][0])):
        US_fugitive_GHG[j] += fugitive_forc_df['Forecasted_fugitive_GHG'][i][j]

# Fourth use calculated growth rates and population denisties to forecast emissions via following tobit model

# \hat{m}_{i,t} = \Phi(\frac{\hat{m}_{i,t}}{\sigma})\ast\hat{m}_{i,t} + \hat{\sigma}\ast\phi(\hat{m}_{i,t})

# Find \sigma as the standard deviation of the residuals from each model

residual_means = np.mean(fugitive_results.resid)
residual_std = np.std(fugitive_results.resid)
        
# Estimate the tobit transformed forecasts

# Performing the tobit transformation on the data

# Transforming the forecasts with the tobit transform

for i in range(len(US_fugitive_GHG)):
    US_fugitive_GHG[i] = tobit_transform(US_fugitive_GHG[i], residual_std)
    
# Lastly, aggregate subsector emissions forecasts and compare to energy emissions forecasts

subsector_aggregated = US_commercial_GHG + US_residential_GHG + US_industrial_GHG + US_transportation_GHG + US_electric_GHG + US_fugitive_GHG
subsector_aggregated = subsector_aggregated / ratio_mean

# Plotting subsector forecasts against estimated forecasts using percentages of full energy sector

# Because of the nature of these plots, this section is abandoned -- we will not use % estiamtes for anything, just use the subsector forecasts

# Plotting subsector forecasts for all subsectors grouped by model type along with corresponding full energy forecast from same model

plt.figure(36)
plt.plot(basis, US_commercial_GHG, label = 'Commercial Subsector', color = cm(30))
plt.plot(basis, US_residential_GHG, label = 'Residential Subsector', color = cm(60))
plt.plot(basis, US_industrial_GHG, label = 'Industrial Subsector', color = cm(90))
plt.plot(basis, US_transportation_GHG, label = 'Transportation Subsector', color = cm(120))
plt.plot(basis, US_electric_GHG, label = 'Electric Power Subsector', color = cm(150))
plt.plot(basis, US_fugitive_GHG, label = 'Fugitive Emissions', color = cm(180))
plt.plot(basis, US_energy_GHG, label = 'Energy Production - Total', color = cm(210))

# Add titles and save
plt.title('Subsector Level GHG Emissions', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 9, ncol = 2)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/subsector_plots.eps')

plt.figure(72)
plt.plot(basis, US_energy_GHG, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/agg_subsector_v_sector.eps')

# Testing the various models for aggregated in-sample data fitting

# Calculate the data mean

Y_bar = [sum(Y_Energy)/len(Y_Energy)]*len(Y_Energy)

# Calculate aggregate predicted values

agg_val = electric_results.fittedvalues + transportation_results.fittedvalues + industrial_results.fittedvalues + commercial_results.fittedvalues + residential_results.fittedvalues + fugitive_results.fittedvalues

# Calculate SSR

agg_sr = (agg_val - Y_bar)**2
agg_ssr = sum(agg_sr)

# Calculate SSTO

agg_sto = (Y_Energy - Y_bar)**2
agg_ssto = sum(agg_sto)

# Calculate $R^{2}$

agg_r2 = agg_ssr / agg_ssto

# Calculate Adjusted $R^{2}$

agg_ar2 = 1 - ((1 - agg_r2) * ((len(Y_Energy) - 1) / (len(Y_Energy) - (energy_results.df_model + 1))))

# Get Adjusted $R^{2}$ from independent energy sector models

ind_ar2 = energy_results.rsquared_adj

# Create dataframe of results and write to file

models = pd.DataFrame(['AR-1 EKC Model'])
ind = pd.DataFrame([ind_ar2])
agg = pd.DataFrame([agg_ar2])
adj_r2_df = pd.concat([models, ind, agg], axis = 1)
adj_r2_df.columns = ['Model', 'Independent', 'Aggregate']
adj_r2_df.to_csv('C:/Users/User/Documents/Data/Regression_Outputs/EKC/adjusted_rsquared.txt', index = False)

plt.figure(80)
base2 = [i for i in range(1991,2012)]
plt.plot(base2, historical, label = 'Historical US Energy Sector GHG Emissions', color = 'black')
plt.plot(basis, US_energy_GHG, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated, label = 'Aggregate of Subsector Forecasts', color = cm(120))

# Add titles and save
plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/forcecasts_with_historical_data.eps')

# Validating the forecast with EPA data

# Loading the EPA Data

epadata = pd.read_csv(epapath)

# Extracting the non-agricultural total GHG emissions from the EPA data and regressing it against WRI historical data

EPA = epadata.Non_Ag[1:len(historical)+1]
EPA = stats.add_constant(EPA)
val_mod = stats.OLS(historical, EPA)
val_res = val_mod.fit()
print(val_res.summary())
file = open('C:/Users/User/Documents/Data/Regression_Outputs/EKC/validation_results.txt', 'w')
file.write(val_res.summary().as_text())
file.close()

# Testing forecasts against future EPA data as a validation measure

# Scaling EPA data per above regression

a = val_res.params['const']
b = val_res.params['Non_Ag']
future_epa = epadata.Non_Ag[len(historical)+1::]
future_epa = future_epa*b + a

# Plots

epaplot = epadata.Non_Ag[1::]*b + a
plt.figure(figsize = (8,5))
plt.ylim(bottom = 4000, top = 6500)
base2 = [i for i in range(1991,2012)]
base3 = [i for i in range(1991,2018)]
plt.plot(base2, historical, label = 'Historical US Energy Sector GHG Emissions', color = 'black')
plt.plot(basis, US_energy_GHG, label = 'Energy Sector Forecast', color = cm(0))
plt.plot(basis, subsector_aggregated, label = 'Aggregate of Subsector Forecasts', color = cm(120))
plt.plot(base3, epaplot, label = 'EPA Data', color = cm(240))

# Add titles and save

plt.title('Comparison of Aggregated Forecasts at the Subsector Level\nand the Forecast for the Energy Sector\n(AR-1 EKC Model)', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('Year')
plt.ylabel('GHG Emissions in Mt CO2 Equivalent')
plt.legend(loc = 8, ncol = 1)
plt.savefig('C:/Users/User/Documents/Data/Regression_Outputs/EKC/forcecasts_with_historical_data_ekc.eps')

# Generating statistics on forecast accuracy

sector = US_energy_GHG[0:6]
subsectors = subsector_aggregated[0:6]

secs = [(future_epa[i+22] - sector[i])**2 for i in range(len(sector))]
subs = [(future_epa[i+22] - subsectors[i])**2 for i in range(len(sector))]

MSE_sector = (1 / len(sector)) * sum(secs)
MSE_subs = (1 / len(sector)) * sum(subs)

print(MSE_sector)
print(MSE_subs)

