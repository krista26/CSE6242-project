# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 20:54:11 2022

@author: krist
"""

import pandas as pd
import numpy as np
import json
import warnings
from dateutil import relativedelta
from dateutil import parser
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima


warnings.filterwarnings("ignore")

temps=pd.read_csv('GlobalLandTemperaturesByCountry.csv', low_memory=False)
emissions=pd.read_csv('co2_emission.csv')
population=pd.read_csv('population_pyramid_1950-2022.csv').drop(columns='Age')
methane=pd.read_csv('methane_hist_emissions.csv').drop(columns=['Gas', 'Unit', 'Sector'])
gdp=pd.read_csv('Countries GDP 1960-2020.csv').drop(columns='Country Code')

#Convert date column to datetime
temps['dt']=pd.to_datetime(temps['dt'])
emissions=emissions.drop(columns='Code')

#Filter out all rows that are from earlier than 1950
temp_data=temps[temps['dt']>'1950-01-01'].reset_index(drop=True)

#There are some duplicate countries, so I am removing those
temp_data=temp_data[~temp_data['Country'].isin(['Denmark (Europe)','France (Europe)','Netherlands (Europe)','United Kingdom (Europe)' ])]
#print(temp_data.Country.unique())

#Adding a year column so we can join CO2 data
temp_data['year'] = pd.DatetimeIndex(temp_data['dt']).year
temp_data['decade']= (pd.DatetimeIndex(temp_data['dt']).year//10)*10


#Adding population columns together then grouping by country and year
population['total']=population['M']+population['F']
population=population.drop(columns=['M','F'])
population=population.groupby(by=['Country', 'Year']).sum().reset_index()
population=population.rename(columns={'Year':'decade'})

population['Country']=population['Country'].replace(['United States of America', 'Russian Federation', 'Viet Nam', 'Venezuela (Bolivarian Republic of)', 'Iran (Islamic Republic of)', 'Republic of Korea', 'United Republic of Tanzania'],
                                                    ['United States', 'Russia', 'Vietnam', 'Venezuela', 'Iran', 'South Korea', 'Tanzania'])
#print(population.Country.unique())
#print(population.head())


#Clean and process methane data
methane=methane.groupby(by='Country').sum()
year_list=list(methane.columns)
methane= pd.melt(methane, value_vars=year_list,value_name='methane', var_name='Year', ignore_index=False).reset_index()
methane['Year']=methane['Year'].astype('int64')

#print(methane.head())


#Clean and process GDP data
gdp=gdp.set_index('Country Name')
year_list=list(gdp.columns)
gdp= pd.melt(gdp, value_vars=year_list,value_name='gdp', var_name='Year', ignore_index=False).reset_index().rename(columns={'Country Name':'Country'})
gdp['Year']=gdp['Year'].astype('int64')
#print(gdp.head())


# %%

#Inner joining so we aren't introducing any countries that aren't contained in both datasets
df=pd.merge(temp_data, emissions, how='inner', left_on=['Country', 'year'], right_on=['Entity', 'Year']).drop(columns=['Entity','year'])
df['Annual CO₂ emissions (tonnes )']=df['Annual CO₂ emissions (tonnes )']/12
#print(new_df)


#Inner joining population data and renaming some columns
df=pd.merge(df, population, how='inner', left_on=['Country', 'decade'], right_on=['Country', 'decade'])
df=df.rename(columns={'Annual CO₂ emissions (tonnes )': 'CO2', 'total':'population'})
#print(df.head())

#print(df['Country'].unique())
#159 countries

#Merging methane data, left merge because we are missing several years compared to other sets
df=pd.merge(df, methane, how='left', left_on=['Country', 'Year'], right_on=['Country', 'Year'])
df['methane']=df['methane']/12


#Joining GDP data w inner join
df=pd.merge(df, gdp, how='inner', left_on=['Country', 'Year'], right_on=['Country', 'Year'])
df['gdp']=df['gdp']/12
df['dt']=df['dt'].astype(str)
print(len(df['Country'].unique()))
#print(df.head(50))
pd.set_option('display.max_columns', None)
#print(df.describe())


result = df.to_json(orient="index")
parsed = json.loads(result)
df_json=json.dumps(parsed, indent=4)[:1000]



countries = df['Country'].unique()
countries2 = ['Lesotho']
countries_df = []
for country in countries:
    df2=df.query("Country == @country")
    countries_df.append(df2)

i=0
tot_rmse = 0
order_array = []
seasonal_array=[]
least_rmse = 1000000000000
for i in range(3):
    for j in range(2):
        for k in range(2):
            order_array.append((i,j,k))



order_array = [(1,0,0),(0,1,0),(1,1,0),(0,1,1),(0,2,1),(0,2,2),(1,1,2)]
order_array = [(0,2,1)]

#trend_array = ['n','c','ct','t']
trend_array = ['t']
ideal_order = (-1,-1,-1)
ideal_trend = 'd'
min_val = 1000000000
end_date = parser.parse("01-01-2050")
end_year = 3000
country_predictions = []
country_changes ={}
for df_temp in countries_df:
    curr_country = df_temp['Country'].unique()[0]

    print(curr_country)
    df_temp['dt'] = pd.to_datetime(df_temp['dt'])
    df_temp_temperature = df_temp.groupby(df_temp['dt'].dt.year)['AverageTemperature'].mean().to_frame()
    df_temp_gdp = df_temp.groupby(df_temp['dt'].dt.year)['gdp'].mean().to_frame()
    df_temp_CO2 = df_temp.groupby(df_temp['dt'].dt.year)['CO2'].mean().to_frame()
    df_temp_methane = df_temp.groupby(df_temp['dt'].dt.year)['methane'].mean().to_frame()
    df_temp_population = df_temp.groupby(df_temp['dt'].dt.year)['population'].mean().to_frame()

    if curr_country != 'Lesotho':
        df_temp_methane=df_temp_methane[30:]
    df_temp_temperature.reset_index(inplace=True)
    df_temp_gdp.reset_index(inplace=True)
    df_temp_CO2.reset_index(inplace=True)
    df_temp_methane.reset_index(inplace=True)
    dates = df_temp_temperature['dt']
    last_date = dates.tail(1).to_list()[0]
    methane_dates = df_temp_methane['dt']
    last_date_methane = methane_dates.tail(1).to_list()[0]
    temperature_data_all = df_temp_temperature['AverageTemperature']
    gdp_data_all = df_temp_gdp['gdp']
    methane_data_all = df_temp_methane['methane']
    CO2_data_all = df_temp_CO2['CO2']
    population_data_all = df_temp_population['population'].unique()

    stepwise_fit_temp = auto_arima(temperature_data_all, trace=False,suppress_warnings=True,with_intercept=True)
    forecast_temp = stepwise_fit_temp.predict(n_periods=(end_year-last_date))
    stepwise_fit_gdp = auto_arima(gdp_data_all, trace=False,suppress_warnings=True,with_intercept=True)
    forecast_gdp = stepwise_fit_gdp.predict(n_periods=(end_year-last_date))
    stepwise_fit_CO2 = auto_arima(CO2_data_all, trace=False,suppress_warnings=True,with_intercept=True)
    forecast_CO2 = stepwise_fit_CO2.predict(n_periods=(end_year-last_date))
    stepwise_fit_methane = auto_arima(methane_data_all, trace=False,suppress_warnings=True,with_intercept=True)
    forecast_methane = stepwise_fit_methane.predict(n_periods=(end_year-last_date))
    stepwise_fit_population = auto_arima(population_data_all, trace=False,suppress_warnings=True,with_intercept=True)
    forecast_population = stepwise_fit_population.predict(n_periods=int(((end_year-last_date)/10)))
    #print(forecast_population)
    #input()

    #print(stepwise_fit.get_params())
    #model = ARIMA(temperature_data_all, order=new_order)
    # model_fit = model.fit()
    #forecast_1 = model_fit.forecast(steps=(end_year-last_date))
    #delta = relativedelta.relativedelta(end_date, last_date)
    #step_num = delta.months + delta.years*12 + 1


    # #forecast_1.to_csv('csv_files/' + curr_country + 'forecast_1.csv')
    # #sarima_rmse = np.sqrt(mean_squared_error(test_data['AverageTemperature'].iloc[:-1], forecast_1))
    # #temp_str = curr_country + ": " + str(sarima_rmse) + '\n'
    # #tot_rmse+=sarima_rmse
    #date_future = pd.date_range(last_date,'2050-01-01', freq='MS').strftime("%Y-%b").tolist()
    date_future = range(last_date,end_year)
    temp_df_stuff = np.column_stack((date_future, forecast_temp, forecast_CO2, forecast_gdp, forecast_methane))
    df_write = pd.DataFrame(temp_df_stuff, columns = ['dt','AverageTemperature','CO2','gdp','methane'])
    df_write['dt'] = df_write['dt'].astype(int)
    df_write ['Country'] = curr_country
    #print(df_write)
    #order_list = str(order_temp)
    percent_change = (df_write.iloc[986]['AverageTemperature'] - df_write.iloc[2]['AverageTemperature'])/df_write.iloc[2]['AverageTemperature']*100
    country_changes[curr_country] = percent_change
    df_write.to_csv('csv_files/' + curr_country + '_forecasts.csv')
    country_predictions.append(df_write)

    #print('Issue with country: ' + curr_country)

for country_pred in country_predictions:
    df = pd.concat([df, country_pred], ignore_index=True)
df.to_csv('new_df.csv')

print(country_changes)
