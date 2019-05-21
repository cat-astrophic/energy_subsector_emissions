%%%%%%%%%% WHAT THIS SCRIPT DOES %%%%%%%%%%

This script uses the accompanying csv files to forecast greenhouse gas (GHG) emissions for the US at the state and national level for the energy production sector and six of its subsectors (seven total categories).

It runs eight different regression models for each of the seven categories: baseline,  year fixed effects, year and state fixed effects, year fixed effects and state random effects, with each of these four models being run with and without a cubic term for gdp per capita.

It then forecasts population density and gdp per capita at the state level for 100 years.

Using these forecasts and the parameters from the regression models, each of these eight  scenarios is forecasted over a 100 year time period for each of the seven different categories.

Forecasts are run again with population density held constant rather than also being forecasted.

A myriad of plots are created and saved as eps files.

Summary outputs from regressions are saved as txt files.

Forecasts are saved as txt and csv files.

%%%%%%%%%% SET UP %%%%%%%%%%

In order to run this script, some initial setup is required.

First, decide where you want to keep the csv data files.

I used the path \Users\User\Documents\Data\

Second, decide where you want to output the txt, csv, and eps files.

I used the path \Users\User\Documents\Data\Regression_Outputs\

Also note that you may change all eps file outputs to png if desired.

%%%%%%%%%% RUNNING THE SCRIPT %%%%%%%%%%

In order to run the script you will need python.

I personally use Anaconda3 with Spyder as an IDE.

You may either run this in your IDE of choice or from the shell with the following command:

C:\Users\User\anaconda3\python.exe C:\Users\User\\.spyder-py3\state_level_stats.py

Note that the first path must be to wherever you have python installed.

The second path directs python to the script itself.

Upon completion the Regression_Outputs folder will contain 237 files.

%%%%%%%%%% CITING THIS PAPER (PREPRINT) %%%%%%%%%%

Bibtex:

@article{cary2019have,

  author = {Cary, Michael},
  
  title = {Have greenhouse gas emissions from US energy production peaked? State level evidence from six subsectors},
  
  journal = {Preprints},
  
  year = {2019}
  
}

%%%%%%%%%% CITING THE DATA SOURCE %%%%%%%%%%

Bibtex:

@article{wri2015data,

  author = {{WRI, CAIT Climate Data Explorer}},
  
  title = {Climate Analysis Indicators Tool: WRI’s Climate Data Explorer},
  
  journal = {Washington, DC: World Resources Institute},
  
  year = {2014},
  
  note = {Available online at: \url{http://cait.wri.org}. Accessed on April 2, 2019}
  
}
