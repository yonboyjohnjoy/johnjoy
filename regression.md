```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col 
import matplotlib.pyplot as plt

df = pd.read_csv('input_data2/housing_train.csv')
summary = pd.DataFrame(columns = ['Beta', 'R2'])




```

## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important

## The Dataset 
This dataset covers the properties of houses & their lots in 1941 of homes between 2006 & 2008.
The majority of entries in this dataset are likely to have missing values, as certain features covered in the dataset (fireplaces, pools, fences) are luxury items that many houses lack. NA values in these categorical varibles may need to be converted to 0 or another indicator value to better analyze them. 


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v_MS_SubClass</th>
      <th>v_Lot_Frontage</th>
      <th>v_Lot_Area</th>
      <th>v_Overall_Qual</th>
      <th>v_Overall_Cond</th>
      <th>v_Year_Built</th>
      <th>v_Year_Remod/Add</th>
      <th>v_Mas_Vnr_Area</th>
      <th>v_BsmtFin_SF_1</th>
      <th>v_BsmtFin_SF_2</th>
      <th>...</th>
      <th>v_Wood_Deck_SF</th>
      <th>v_Open_Porch_SF</th>
      <th>v_Enclosed_Porch</th>
      <th>v_3Ssn_Porch</th>
      <th>v_Screen_Porch</th>
      <th>v_Pool_Area</th>
      <th>v_Misc_Val</th>
      <th>v_Mo_Sold</th>
      <th>v_Yr_Sold</th>
      <th>v_SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1941.000000</td>
      <td>1620.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1923.000000</td>
      <td>1940.000000</td>
      <td>1940.000000</td>
      <td>...</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>58.088614</td>
      <td>69.301235</td>
      <td>10284.770222</td>
      <td>6.113344</td>
      <td>5.568264</td>
      <td>1971.321999</td>
      <td>1984.073158</td>
      <td>104.846074</td>
      <td>436.986598</td>
      <td>49.247938</td>
      <td>...</td>
      <td>92.458011</td>
      <td>49.157135</td>
      <td>22.947965</td>
      <td>2.249871</td>
      <td>16.249871</td>
      <td>3.386399</td>
      <td>52.553838</td>
      <td>6.431221</td>
      <td>2006.998454</td>
      <td>182033.238022</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.946015</td>
      <td>23.978101</td>
      <td>7832.295527</td>
      <td>1.401594</td>
      <td>1.087465</td>
      <td>30.209933</td>
      <td>20.837338</td>
      <td>184.982611</td>
      <td>457.815715</td>
      <td>169.555232</td>
      <td>...</td>
      <td>127.020523</td>
      <td>70.296277</td>
      <td>65.249307</td>
      <td>22.416832</td>
      <td>56.748086</td>
      <td>43.695267</td>
      <td>616.064459</td>
      <td>2.745199</td>
      <td>0.801736</td>
      <td>80407.100395</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1470.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>13100.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7420.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.000000</td>
      <td>1965.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2006.000000</td>
      <td>130000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9450.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>361.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2007.000000</td>
      <td>161900.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11631.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>168.000000</td>
      <td>735.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2008.000000</td>
      <td>215000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>164660.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2008.000000</td>
      <td>2009.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>1424.000000</td>
      <td>742.000000</td>
      <td>1012.000000</td>
      <td>407.000000</td>
      <td>576.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2008.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>



As stated previously, there are a large number of categorical/ordinal varibles that need to be converted to ints for proper evaulation.

## Distribution of Sale Prices


```python
sns.set_theme(style="whitegrid")
ax = sns.distplot(x=df["v_SalePrice"]).set(title='Distribution of Sale Prices')
```

    /Users/johnjoy/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_6_1.png)
    


## Average Sale Price by year


```python
df.groupby("v_Yr_Sold")["v_SalePrice"].mean()


```




    v_Yr_Sold
    2006    181761.648000
    2007    185138.207493
    2008    178841.750804
    Name: v_SalePrice, dtype: float64



## Correlation with Lot Area


```python
 df['v_SalePrice'].corr(df['v_Lot_Area'])

```




    0.2580274833330576



## Correlation with Year Sold


```python
df['v_SalePrice'].corr(df['v_Yr_Sold'])
```




    -0.014515613107063187



## Correlation with aboveground living area


```python
df['v_SalePrice'].corr(df['v_Gr_Liv_Area'])
```




    0.7159200111944868



## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Yr_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v_Yr_Sold==2007})+ \beta_2 * (\text{v_Yr_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? I won't be shocked if someone beats that!
    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.


## Sale Price to Lot Area Regression


```python
df = pd.read_csv('input_data2/housing_train.csv')
reg1 = sm_ols('v_SalePrice ~  v_Lot_Area ', data=df).fit()
reg1.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>v_SalePrice</td>   <th>  R-squared:         </th> <td>   0.067</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.066</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   138.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th> <td>6.82e-31</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -24610.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>4.922e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1939</td>      <th>  BIC:               </th> <td>4.924e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td> 1.548e+05</td> <td> 2911.591</td> <td>   53.163</td> <td> 0.000</td> <td> 1.49e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>v_Lot_Area</th> <td>    2.6489</td> <td>    0.225</td> <td>   11.760</td> <td> 0.000</td> <td>    2.207</td> <td>    3.091</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>668.513</td> <th>  Durbin-Watson:     </th> <td>   1.064</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3001.894</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.595</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.191</td>  <th>  Cond. No.          </th> <td>2.13e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.13e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Sale Price to log of Lot Area Regression


```python
df = (df.assign(l_Lot_Area = np.log(df['v_Lot_Area'])))
reg2 = sm_ols('v_SalePrice ~  l_Lot_Area ', data=df).fit()
reg2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>v_SalePrice</td>   <th>  R-squared:         </th> <td>   0.128</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.128</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   285.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th> <td>6.95e-60</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -24544.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>4.909e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1939</td>      <th>  BIC:               </th> <td>4.910e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>-3.279e+05</td> <td> 3.02e+04</td> <td>  -10.850</td> <td> 0.000</td> <td>-3.87e+05</td> <td>-2.69e+05</td>
</tr>
<tr>
  <th>l_Lot_Area</th> <td> 5.603e+04</td> <td> 3315.139</td> <td>   16.901</td> <td> 0.000</td> <td> 4.95e+04</td> <td> 6.25e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>650.067</td> <th>  Durbin-Watson:     </th> <td>   1.042</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2623.687</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.587</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.729</td>  <th>  Cond. No.          </th> <td>    164.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Log of Sale Price to Lot Area Regression


```python
df = (df.assign(l_SalePrice = np.log(df['v_SalePrice'])))
reg3 = sm_ols('l_SalePrice ~  v_Lot_Area ', data=df).fit()
reg3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>l_SalePrice</td>   <th>  R-squared:         </th> <td>   0.065</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.064</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   133.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th> <td>5.46e-30</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -927.19</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>   1858.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1939</td>      <th>  BIC:               </th> <td>   1870.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>   11.8941</td> <td>    0.015</td> <td>  813.211</td> <td> 0.000</td> <td>   11.865</td> <td>   11.923</td>
</tr>
<tr>
  <th>v_Lot_Area</th> <td> 1.309e-05</td> <td> 1.13e-06</td> <td>   11.571</td> <td> 0.000</td> <td> 1.09e-05</td> <td> 1.53e-05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>75.460</td> <th>  Durbin-Watson:     </th> <td>   0.980</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 218.556</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.066</td> <th>  Prob(JB):          </th> <td>3.48e-48</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.639</td> <th>  Cond. No.          </th> <td>2.13e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.13e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Log of Sale Price to Log of Lot Area


```python
reg4 = sm_ols('l_SalePrice ~  l_Lot_Area ', data=df).fit()
reg4.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>l_SalePrice</td>   <th>  R-squared:         </th> <td>   0.135</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.135</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   302.5</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th> <td>4.38e-63</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -851.27</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>   1707.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1939</td>      <th>  BIC:               </th> <td>   1718.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>    9.4051</td> <td>    0.151</td> <td>   62.253</td> <td> 0.000</td> <td>    9.109</td> <td>    9.701</td>
</tr>
<tr>
  <th>l_Lot_Area</th> <td>    0.2883</td> <td>    0.017</td> <td>   17.394</td> <td> 0.000</td> <td>    0.256</td> <td>    0.321</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>84.067</td> <th>  Durbin-Watson:     </th> <td>   0.955</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 255.283</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.100</td> <th>  Prob(JB):          </th> <td>3.68e-56</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.765</td> <th>  Cond. No.          </th> <td>    164.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Log of Sale Price to year sold


```python
reg5 = sm_ols('l_SalePrice ~  v_Yr_Sold ', data=df).fit()
reg5.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>l_SalePrice</td>   <th>  R-squared:         </th> <td>   0.000</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.000</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.2003</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th>  <td> 0.655</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -991.88</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>   1988.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1939</td>      <th>  BIC:               </th> <td>   1999.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   22.2932</td> <td>   22.937</td> <td>    0.972</td> <td> 0.331</td> <td>  -22.690</td> <td>   67.277</td>
</tr>
<tr>
  <th>v_Yr_Sold</th> <td>   -0.0051</td> <td>    0.011</td> <td>   -0.448</td> <td> 0.655</td> <td>   -0.028</td> <td>    0.017</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>55.641</td> <th>  Durbin-Watson:     </th> <td>   0.985</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 131.833</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.075</td> <th>  Prob(JB):          </th> <td>2.36e-29</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.268</td> <th>  Cond. No.          </th> <td>5.03e+06</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 5.03e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Log of Sale Price to Yeard Sold in 2007 & 2008


```python
reg6 = sm_ols ('l_SalePrice ~  v_Yr_Sold==2007 + v_Yr_Sold==2008', data=df).fit()
reg6.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>l_SalePrice</td>   <th>  R-squared:         </th> <td>   0.001</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.000</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.394</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th>  <td> 0.248</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -990.59</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>   1987.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1938</td>      <th>  BIC:               </th> <td>   2004.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>   12.0229</td> <td>    0.016</td> <td>  745.087</td> <td> 0.000</td> <td>   11.991</td> <td>   12.055</td>
</tr>
<tr>
  <th>v_Yr_Sold == 2007[T.True]</th> <td>    0.0256</td> <td>    0.022</td> <td>    1.150</td> <td> 0.250</td> <td>   -0.018</td> <td>    0.069</td>
</tr>
<tr>
  <th>v_Yr_Sold == 2008[T.True]</th> <td>   -0.0103</td> <td>    0.023</td> <td>   -0.450</td> <td> 0.653</td> <td>   -0.055</td> <td>    0.035</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>54.618</td> <th>  Durbin-Watson:     </th> <td>   0.989</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 127.342</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.080</td> <th>  Prob(JB):          </th> <td>2.23e-28</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.245</td> <th>  Cond. No.          </th> <td>    3.79</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Log of Sale price to 1st floor square feet, second floor square feet, low quality finshed square feet, above ground living area, size of garage,  & value of misc features 


```python


reg7 = sm_ols ('l_SalePrice ~ v_Lot_Area + v_1st_Flr_SF + v_2nd_Flr_SF + v_Low_Qual_Fin_SF + v_Gr_Liv_Area + v_Misc_Val',  data=df).fit()
reg7.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>l_SalePrice</td>   <th>  R-squared:         </th> <td>   0.568</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.567</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   508.2</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 03 Apr 2022</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:35</td>     <th>  Log-Likelihood:    </th> <td> -178.11</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>   368.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1935</td>      <th>  BIC:               </th> <td>   401.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   11.0547</td> <td>    0.021</td> <td>  534.447</td> <td> 0.000</td> <td>   11.014</td> <td>   11.095</td>
</tr>
<tr>
  <th>v_Lot_Area</th>        <td>-6.978e-07</td> <td> 8.33e-07</td> <td>   -0.838</td> <td> 0.402</td> <td>-2.33e-06</td> <td> 9.36e-07</td>
</tr>
<tr>
  <th>v_1st_Flr_SF</th>      <td>    0.0005</td> <td>  3.7e-05</td> <td>   14.455</td> <td> 0.000</td> <td>    0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>v_2nd_Flr_SF</th>      <td>    0.0003</td> <td> 3.71e-05</td> <td>    6.746</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>v_Low_Qual_Fin_SF</th> <td>   -0.0006</td> <td>    0.000</td> <td>   -5.671</td> <td> 0.000</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>v_Gr_Liv_Area</th>     <td>    0.0002</td> <td> 3.56e-05</td> <td>    5.230</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>v_Misc_Val</th>        <td>-6.234e-05</td> <td> 9.88e-06</td> <td>   -6.307</td> <td> 0.000</td> <td>-8.17e-05</td> <td> -4.3e-05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>845.073</td> <th>  Durbin-Watson:     </th> <td>   1.346</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>10600.345</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.700</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.932</td>  <th>  Cond. No.          </th> <td>2.46e+17</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 5.43e-24. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 


# Betas 

Reg 1 Beta : 2.6489
Reg 2 Beta : 5.603e+4
Reg 3 Beta :1.309e-4
Reg 4 Beta: .2883
Reg 5 Beta: -.0051
Reg 6 Beta 1: 0256 Beta 2: -.103


## 2. Model 2 Beta 1 interpretation - a 1% increase in lot area results in a $560 increase in sale price

## 3. Model 3 Beta 1 interpretation - a 11 square foot change in area results in a 1% change in sale price

## 4. I think Model 4 is the best, as it compares a normalized version of each value

## 5. Model 5 B1 interpretation - A 1 unit increase in year has results in a 5.1% decrease in share price

## 6. The average log of sale price for houses sold in 2007 & 2008 is 12.0229

## Houses sold for 2.56% more in 2007 than in 2008

## It's higher because there are multiple variables used in the regression, and R^2 increases with number of variables

## I used  1st floor square feet, second floor square feet, low quality finshed square feet, above ground living area, size of garage,  & value of misc features 

## My R^2 was .568

## I don't think you could use model 6 in a predictive regression as you'd need the average house price sale in 2007 & 2008.

## You may be able to use model 5 in a predictive regression but I question it's accurary.


```python

```
