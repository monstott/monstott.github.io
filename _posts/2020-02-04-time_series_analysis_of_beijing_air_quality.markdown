---
layout: post
title:      "Time Series Analysis of Beijing Air Quality"
date:       2020-02-04 07:44:47 +0000
permalink:  time_series_analysis_of_beijing_air_quality
---


## Motivation for this post.
Time series analysis is the application of methods for analyzing time series data to efforts extracting meaningful statistics and characteristics. A time series is a collection of data points that possess a natural temporal ordering. Time series are usually sequenced at successive equally-spaced points in time, making them discrete. 

Applying time series analysis to available data on airborne particulates may provide insight on future health impacts in a particular area. Uncovered statistics and prediction forecasts from the analysis are, at the very least, likely to contain interesting information. The goal of this post is to assess the presence of particles over time that are less than 10 micrometers in size within the Beijing area. Particles in this size range are capable of being drawn into the lungs. In turn, particles in the lungs can be absorbed into the blood and have a direct physical effect on the body. It should be stressed, however, that health effects from particulates are dependent on more than just their size. Other important properties include particle chemical composition, physical properties, mass concentration, and duration of exposure. More information on particulate matter in this size range can be found [here](http://www.npi.gov.au/resource/particulate-matter-pm10-and-pm25). 

Given the information available in the dataset, this analysis will be able to comment on the *potential* risk to health, on general air quality and on visibility. 

## The dataset.
The dataset  I will be working with is the [Beijing Multi-Site Air-Quality Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data#) from the University of California-Irvine (UCI) [Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). This dataset contains hourly air pollutants data from the Beijing Municipal Environmental Monitoring Center for 12 nationally-controlled air-quality monitoring sites. Meteorological data for each air-quality site is matched with the nearest China Meteorological Administration weather station so that conditions at the time of each observation are recorded. The 12 sites are Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, and Wanshouxigong. The data for each site is located within its own CSV file. 

The site I will be concerned with for this investigation is Shunyi. Shunyi is an administrative district located to the northeast of the Beijing's urban core. The Shunyi dataset is comprised of 35,064 records, each with 18 attributes. 

## Exploratory data analysis.
```
# import data
import pandas as pd

df = pd.read_csv('./Blog3/PRSA_Data_Shunyi_20130301-20170228.csv')
df.shape

> (35064, 18)
```

Time information is stored in four separate columns for year, month, day, and hour. These columns need to be combined into a datetime object that will then serve as the index.

The time range of the dataset is from midnight on March 1st, 2013 to 11 pm on February 28th, 2017. 

```
# convert to datetime index
df.index = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
print(df.index[0])
print(df.index[-1])

> 2013-03-01 00:00:00
> 2017-02-28 23:00:00
```

Of the 18 columns in the dataset, the one I will be concerned with is **PM10**. This column reports the concentration (in micrograms per cubic meter) of particulate matter 10 micrometers or less in diameter.  

```
# view columns
df.columns

> Index(['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
       'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station'],
      dtype='object')
```

The mean **PM10** value in this dataset from 2013 to 2017 is 99  μg / m<sup>3</sup>. The standard deviation is high at 89 μg / m<sup>3</sup>. It follows then that the range is large: the minimum value is 2 μg / m<sup>3</sup> and the maximum value is 999 μg / m<sup>3</sup>. 

```
# descriptive stats
df.PM10.describe()

> count    34516.000000
> mean        98.737026
> std         89.143718
> min          2.000000
> 25%         31.000000
>50%         77.000000
> 75%        138.000000
> max        999.000000
> Name: PM10, dtype: float64
```
The **PM10** column has 548 rows of missing data, about 1.6% of the dataset. 

```
# missing values
df.isna().sum()

> No       0
> year     0
> month    0
> day      0
> hour     0
> PM2.5    913
> PM10     548
> SO2      1296
> NO2      1365
> CO       2178
> O3       1489
> TEMP     51
> PRES     51
> DEWP     54
> RAIN     51
> wd       483
> WSPM     44
> station  0
> dtype: int64
```

Realistically, the ability to respond to hourly updates and affect meaningful change is low. The dataset frequency is high and the values it contains are susceptible to high variation and noise. If, for example, the result of this investigation is to enact better pollution controls then the dataset should be adjusted to a timescale more amenable to human reaction. 

In line with this thinking, the dataset will be changed to reflect weekly averages. It is important to note that the `asfreq()` method is a filter function that returns the last observation of the desired time grouping and will not perform the desired action. The `resample()` method chained with `mean()` will return the data in the correct form.

```
# convert frequency to weeks
df_wk = df.resample('W').mean()
print('week frequency shape:', df_wk.shape)
print('PM10 missing values:', df_wk.isna().sum()['PM10'])

> week frequency shape: (210, 16)
> PM10 missing values: 0
```

As expeced, this aggregation has removed the missing values. This transformation has also reduced the dataset from 35,064 to 210 observations, a reduction of 99.4%. In another context this might cause an organization to collect more data (or change frequency). The sample size decrease does not, however, impact the demonstration of time series analysis skills. The next action is to assess variation.  
## Trends and repeating variations.
One of the most important features of a time series is its variation. A time series that has patterns repeating over known and fixed periods of time is said to display seasonality. Seasonality is a general term for types of variation that fall into 3 categories.

**Seasonality Components:**

* **Trend:** general, long-term, average tendencies of the data to increase or decrease over time. 
* **Seasonal:** the variations in a time series that operate in a regular and periodic manner over a span of time. If the time span is more than one year the variation is said to be cyclic (instead of seasonal).
* **Random:** the variations of a time series that are irregular or erratic.

```
mport matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plt.plot(df.PM10)
plt.title('Beijing Air Quality at Shunyi Site: PM10 Concentration from 2013 - 2017', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('PM10 Concentration (ug / m^3)', fontsize=14);
```

![Shunyi PM10 2013 to 2017](https://github.com/monstott/Blogs/raw/master/Blog3/Shunyi2013to2017week.png)

**Observations:**
* There are no obvious patterns in the **PM10** time series plot.
* There does not appear to be a general trend increasing or decreasing.
* There may be seasonal variation. There is a slight sinusoidal rising and falling of values that traces across the years. 
* Most of the movement seems random. As an example, the peaks at the start of 2016 and 2017 are not found in earlier years.

I'll take another look at finding patterns within the data using the `seasonal_decompose()` method form the `statsmodels` package. 

```
# time series decomposition
import statsmodels.api as sm
plt.style.use('seaborn-dark-palette')
plt.rcParams['figure.figsize'] = (12, 6)

decomposition = sm.tsa.seasonal_decompose(df_wk.PM10, model='additive', extrapolate_trend='freq')
fig = decomposition.plot()
```

![Decomposition](https://github.com/monstott/Blogs/raw/master/Blog3/decomposition.png)
​

**Observations:**
* The *Observed* plot shows the average weekly **PM10** values.
* The *Trend* plot indicates that there is an overall decreasing trend. The dataset starts at values around 95 and (after increasing in 2014) ends just below 90.
* The *Seasonal* plot identifies repeating patterns that reach their lowest value about 60% of the way through each year. The maximum values are near the start of each year.
* The *Residual* plot reflects the remaining noise in the dataset after removing the other variation types. There are no patterns present.

## Series stationarity.
Next, I'll look at the stationarity of the data. A stationary process has a mean, variance and autocorrelation structure that do not change over time. Stationarity is an assumption made by many time series analysis techniques. The augmented Dickey-Fuller test is used to check for stationarity. It has a null hypothesis that a problematic feature is present in the process (non-stationary). The alternative hypothesis is that the problematic feature is absent and the process is stationary.

```
# augmented dickey-fuller test
from statsmodels.tsa.stattools import adfuller

df_aft = adfuller(df_wk.PM10, autolag='AIC')
output = pd.Series(df_aft[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of observations'])
output

> test statistic             -4.838580
> pvalue                      0.000046
> number of lags used         3.000000
> number of observations    206.000000
> dtype: float64
```

The p-value for the air quality data is less than the significance level of 0.05. The null hypothesis is rejected in favor of the alternative hypothesis which states that the dataset is stationary and its time series properties are constant over time.

## Training and testing sets.
The next step is to split the dataset into training and testing sets. The goal is to build a model that forecasts well on unseen data. This action prevents points in time that I'm interested in predicting from leaking information into the modeling process.

```
# train-test split
pct_train = 0.80
split_wk_idx = round(len(df_wk) * pct_train)
train_wk, test_wk = df_wk[:split_wk_idx], df_wk[split_wk_idx:]
```

## Correlation coefficients.
I will now look at plots of the autocorrelation coefficient and partial autocorrelation coefficient.

* An auto-correlation function (ACF) Plot shows serial correlation in data that changes over time. Serial correlation occurs when an error at one point in time is present at a subsequent point in time. An ACF plot provides a summary of correlation at different periods of time. The plot shows the correlation coefficient on the y-axis for the time series lagged by periods of delay on the x-axis. As an example, since this dataset is indexed by week, at x = 1 the `2013-03-03` record is compared with the `2013-03-10` record; at x = 2, it is compared with both the `2013-03-17` and the `2013-03-10` records. 

* A partial auto-correlation (PACF) plot provides partial autocorrelation coefficients at lags across the x-axis, controlling for previous lags at each subsequent point. The partial autocorrelation at lag 10 is the autocorrelation between the time series and the time series with a lag of 10 that has not already been accounted for by lags 1 through 9. 

```
from statsmodels.tsa.stattools import acf, pacf
import numpy as np

# ACF
lag_acf = acf(train_wk.PM10, nlags=20)

plt.figure(figsize=(12, 10))
plt.subplot(211)
plt.stem(lag_acf)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(train_wk)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_wk)), linestyle='--', color='gray')
plt.title('ACF Plot: PM10 Concentration', fontsize=18)
plt.xlabel('Lag', fontsize=14)
plt.ylabel('Autocorrelation', fontsize=14)

# PACF
lag_pacf = pacf(train_wk.PM10, nlags=20, method='ols')

plt.subplot(212)
plt.stem(lag_pacf)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(train_wk)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_wk)), linestyle='--', color='gray')
plt.title('PACF Plot: PM10 Concentration', fontsize=18)
plt.xlabel('Lag', fontsize=14)
plt.ylabel('Partial Autocorrelation', fontsize=14)

plt.tight_layout();
```

![ACF-PACF Plots](https://github.com/monstott/Blogs/raw/master/Blog3/ACF_PACFweek.png)

**Observations:**
* The ACF plot and PACF plots do not possess high correlation coefficients that would be a cause for concern going into the modeling process. 
* All of the lag values have coefficients within the acceptable levels marked with dashed grey lines.

## SARIMA model.
The model used to forecast the time series is the **S**easonal **A**uto-**R**egressive **I**ntegrated **M**oving **A**verage (SARIMA) model. The parameters for this model can be divided into two groups, the seasonal parameters (S) and the non-seasonal parameters (ARIMA).

### Non-Seasonal Parameters:
There are 3 parameters in an ARIMA model that together model the variation in a time series.

**ARIMA Terms:**

* *p*: the order of the auto-regressive (AR) model (i.e., the number of lagged observations).
* *d*: the degree of differencing required to make the time series stationary.
* *q*: the order of the moving average (MA) model (i.e., the size of the window function over the time series).

ARIMA models are written in the form ARIMA(p, d, q). 

A model with no differencing term is considered an ARMA model. The building blocks for all three models (ARMA, ARIMA, and SARIMA) are the auto-regressive model (AR) and the moving average (MA) model. The AR model forecasts a variable using a linear combination of its previous values. Similarly, the MA model forecasts a variable using a linear combination of its previous error terms. The integration term (I) expands the use case of these two building blocks to non-stationary data.

**Differencing Parameter:**
* The *d* parameter adjusts the ARIMA model for non-stationarity. Stationary time series are easier to forecast.
* The value of *d* is the number of times that differences are computed. For example, if *d* = 2, the procedure calculates the difference twice to determine the result. By comparison, lag is an offset in time periods.

### Seasonal Parameters:
Seasonality is important  to include in models for time series that possess trends that depend on the time of year. The parameters *P*, *D*, and *Q* describe the same associations as *p*, *d*, and *q*, but for the seasonal components of the model.
 
 **Seasonal Terms:** 

* *P*: the order of the seasonal auto-regressive (AR) model.
* *D*: the degree of seasonal differencing applied to the time series.
* *Q*: the order of the seasonal moving average (MA) model.
*  *m*: the number of observations per seasonal cycle (i.e., model seasonality).

**Seasonality Periods:**
* m = 4: quarterly
* m = 12: monthly 
* m = 52: weekly
* m = 365: daily

SARIMA models are written in the form SARIMA(p, d, q)x( P, D, Q, m).

### Model Selection:

The `auto_arima()`  function of the `pyramid` package fits the best SARIMA model to a one-variable time series according to a performance criterion (e.g., AIC). The function performs a search over possible non-seasonal and seasonal parameters within the constraints provided by the user. It selects the parameters that minimize the chosen performance metric. One of the most common metrics used is the Akaike Information Criterion (AIC).

**Akaike Information Criterion:**
* This information criteron estimates the relative quality of statistical models for a given set of data.
* AIC takes into account the complexity of a model along with how well a model fits the data. A model with fewer features will receive a lower AIC score than a similar model with more features to indicate it is the better choice.

I'll build a model with a seasonality that reflects the number of weeks in a year, *m* = 52. Since the parameters are not known, a grid search is conducted to find them. Arguments that define the search boundaries are provided to the method; `auto_arima()` cycles through the parameter values in a stepwise fashion, comparing the AIC scores of the models built from them. In the end, the fit of the best model to the training set is returned. 

This search will start at zero for the non-differencing terms. The maximum parameter values are set artificially high to guarantee a thorough search. 

```
# fit SARIMA model
import pmdarima as pm

fit_wk = pm.auto_arima(train_wk.PM10, start_p=0, d=1, start_q=0, max_p=6, max_d=6, max_q=6, 
                             start_P=0, D=1, start_Q=0, max_P=6, max_D=6, max_Q=6, seasonal=True, m=52, trace=True,
                             error_action='ignore', suppress_warnings=True, stepwise=True)  

fit_wk.summary()
```

![Model Summary](https://github.com/monstott/Blogs/raw/master/Blog3/sarima.PNG)

**Observations:**
* The model summary depicts the impact of each variables on the forecast.
* There are four main lagged AR and MA variables:
 * The first AR  variable *ar.L1* is lagged by 1 time step.
 * The first MA variable *ma.L1* is lagged by 1 time step.
 * The second AR variable *ar.L2* is lagged is lagged by 2 time steps.
 * The third AR variable *ar.S.L52* is a seasonal term and is lagged by 52 time steps.
* Nearly all of the p-values of the variable coefficients are above the standard significance level of 0.05. 
 * The one variable that is statistically significant (p < 0.05) is the seasonal *ar.S.L52* term.
* The performance metric used to evaluate all models is minimized at a value of AIC = 1,253. 


### Model Prediction:
The model summary provides a great level of detail but it doesn't describe exactly when the model performs well or at what times it fails to capture time series variation. The next action is to visualize how well the model fits the data. The best model of the training data will be fit on the testing data and then all observations will be compared against their predictions.

```
# prediction plot
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.set_style('ticks')

plt.scatter(train_wk.index, train_wk.PM10, color='steelblue', marker='o')
plt.plot(train_wk.index, fit_wk.predict_in_sample(), color='steelblue', linewidth=3, alpha=0.6)

fit_test, ci_test = fit_wk.predict(n_periods=test_wk.shape[0], return_conf_int=True)
ci_lower = pd.Series(ci_test[:, 0], index=test_wk.index)
ci_upper = pd.Series(ci_test[:, 1], index=test_wk.index)
plt.scatter(test_wk.index, test_wk.PM10, color='darkred', marker='D')
plt.plot(test_wk.index, fit_wk.predict(n_periods=test_wk.shape[0]), color='darkred', linestyle='--', linewidth=3, alpha=0.6)

plt.title('SARIMA Forecast of Beijing Air Quality', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('PM10 Concentration `(ug / m^3)', fontsize=14)
plt.axvline(x=df_wk.PM10.index[split_wk_idx], color='black', linewidth=4, alpha=0.4)
plt.fill_between(ci_lower.index, ci_lower, ci_upper, color='k', alpha=0.2)
plt.legend(('Data', 'Forecast', '95% Confidence Interval'), loc='best', prop={'size': 12})
plt.show();
```

![Forecast Fit](https://github.com/monstott/Blogs/raw/master/Blog3/forecast2.png)

**Observations:**
* The best SARIMA model does a decent job fitting both the training set and the testing set.
 * All of the test set observations are within the 95% confidence interval, except for one outlier at `2017-01`.
* There are issues, however:
 * Near `2014-03` the model overestimates the **PM10** concentration.
 * Around `2016-01` the model underestimates the **PM10** concentration.
 * Generally speaking, the predictions for the test set are low. There is a risk that this model would provide inaccurate optimistic estimates of particulate concentration if put into use.

### Model Performance:
The last step is to quantify the model performance. The root mean square error (RMSE) is the square root of the variance of the residuals. RMSE is a good measure for determining how well a model predicts data. Since the units of this statistic are the same as the response variable, the interpretation of the model fit is simplified.

```
# root mean square error 
from sklearn.metrics import mean_squared_error

print('Training RMSE: %.2f' % np.sqrt(mean_squared_error(train_wk.PM10, fit_wk.predict_in_sample())))
print('Testing RMSE: %.2f' % np.sqrt(mean_squared_error(test_wk.PM10, fit_wk.predict(n_periods=test_wk.shape[0]))))

> Training RMSE: 51.26
> Testing RMSE: 55.64
```

The RMSE of the testing set (56 μg / m<sup>3</sup>) is nearly equal to the RMSE of the training set (51 μg / m<sup>3</sup>). This means that the model did not overfit to the training data. If it did, there would be a large gap between these values. Given that the standard deviation of **PM10** is 89 μg / m<sup>3</sup>, the average deviation from the SARIMA model is within one standard deviation of the response variable distribution. This translates to solid performance and forecasts with predictions close to the actual observations.

## Final thoughts.
This time series analysis investigated the concentration of particles less than 10 micrometers in size in urban Beijing air. Hourly raw data was transformed into weekly averages to form a manageable dataset capable of providing results at an actionable level. Time series decomposition revealed that the concentration of particles is decreasing in the long term. In the short term, there are oscillations in the particle concentration with troughs near the middle of the year and crests at the start of the year. The assumptions of stationarity and autocorrelation were next assessed before moving into the modeling phase. 

In the modeling phase, the SARIMA model was reviewed and a grid search of parameters was undertaken to find a premium fit of the training set. The final model was then forecast across the same time as the testing set and its predictions compared with the observations. Results indicate that the final model forecast tends to underestimate values. The deviation is within acceptable limits for the test set but is likely to increase over time. 

All things considered, this model provides an excellent approximation of particulate matter at the Shunyi site. It also sheds light on the trends, long-term and seasonal, that are otherwise impossible to pick out of the raw data. It appears that the potential risk to health in the area is decreasing and the general air quality is increasing. At the very least, visiblity has improved. In a very real sense, things are looking sunny in Beijing.

 
