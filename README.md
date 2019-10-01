## Description

This script uses the accompanying csv files to forecast greenhouse gas (GHG) emissions for the US at the state and national level for the energy production sector and six of its subsectors (seven total categories).

It runs two different regression models for each of the seven categories: both are AR-1 autoregressive models, one specified to test the EKC hypothesis, the other is a linear GDP model

It then forecasts gdp per capita and a series of covariates at the state level.

Using these forecasts and the parameters from the regression models, each of these scenarios is forecasted for each of the seven different categories.

A few plots are created and saved as eps files.

Summary outputs from regressions are saved as txt files.

Forecasts are saved as txt and csv files.

## Setup

In order to run this script, some initial setup is required.

First, decide where you want to keep the csv data files.

I used the path \Users\User\Documents\Data\

Second, decide where you want to output the txt, csv, and eps files.

I used the path \Users\User\Documents\Data\Regression_Outputs\

There are two subdirectories which need to be created (eventually this will be automated, but resubmission deadlines and all...)

\Users\User\Documents\Data\Regression_Outputs\EKC\

and

\Users\User\Documents\Data\Regression_Outputs\Linear\

Also note that you may change all eps file outputs to png if desired.

## Citation

Bibtex:

@article{cary2019emissions,\
&nbsp;&nbsp;&nbsp;&nbsp;author = {Cary, Michael},\
&nbsp;&nbsp;&nbsp;&nbsp;title = {Have greenhouse gas emissions from US energy production peaked? State level evidence from six subsectors},\
&nbsp;&nbsp;&nbsp;&nbsp;journal = {Preprints},\
&nbsp;&nbsp;&nbsp;&nbsp;year = {2019}\
}

Don't forget to cite any data sources you might use independently of this! (see paper for details)
