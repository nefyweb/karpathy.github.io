

#### Earnings Suprise Prediction Model (FPM)

This model has been applied to earnings suprises, but would easily be transferable to predicting any unexpected financial event. 

#### Description

Predicting Surprises: The Application of Machine Learning Ensembles in Predicting Earnings Surprises.

1. Introduction and Research 

It is notoriously difficult to predict financial events and outcomes. Researchers and financial institutions often develop algorithms to predict the financial performance and outcomes of firms. Successful predictions are highly consequential: if used correctly it can lead to profitable trading strategies. Having the ability to better predict financial results can help with purchase, sell and hedging decisions. This paper focuses on the task of predicting future earnings surprises for firms in the S&P 500. This study focuses on machine learning methods since data relating to stocks can be categorised as non-stationary time series data allowing for interesting patterns and relationships to be uncovered. This study will focus on non-linear machine learning procedures to predict earnings surprises. In recent years’ numerous innovations in the field of machine learning allows for more accurate and generalizable results such as the commoditisation of deep learning techniques, advancements in GPU web services for parallel computing and model specific research.  To reasonably evaluate the proposal for this study I have trained initial models with attached results. 

For this study I will, as outlined by Kuhn and Johnson (2013), select the best model by starting with the most flexible models after which I will investigate less opaque models and consider whether it reasonably approximates the performance of the more flexible and complex models. Investigating a suite of complex models helps to establish a performance ceiling. My preliminary results indicate that the more complex models do outperform. In finance research, the interpretability of empirical results performed on the basis of proving a theoretical hypothesis, often calls for causality. In this study, accuracy trumps the interpretability of the model, the study does not attempt to show the causality of previously uncovered features in explaining the result of an earnings surprise. Instead, the study focuses on the use of machine learning and the incorporation of a large bag of features in predicting financial outcomes such as earnings surprises. Ensemble models and deep learning models are very difficult to understand even by those who create them, this should, however, not stop us from using these models to identify their practical value in a field.

Trained models can be verified by splitting the sample into a test and training set. To prevent overfitting, the final results will be calculated on an ‘out of sample’ holdout set to predict a generalizable accuracy score. If data-availability becomes an issue, a researcher can also employ cross-validation steps. The data used in the preliminary study stretches from 2005 to 2016, roughly allowing for 10 years of earnings data for all firms. The initial dataset has a little less than 20,000 quarterly earnings samples labelled by whether the estimates has been beaten (1) or whether it fell short to the actuals (0). 60% of the sample data is used to train the model. Due to the time relevance of the data, training over long periods might suffocate the future generalisability of the functional approximations, therefore, any increase in sample size should rather occur cross-sectionally. 

Whether a company ‘misses’ or ‘beats’ expectations can have a substantial price impact, both immediately and over the course of the following days/weeks (Jones et al. 1970). Analysts and investors can succumb to many behavioural biases and conflicts of interest, research show that investors and analysts over extrapolate from past earnings (Givly et al, 1984). This study would circumvent biased decision-making by introducing a bag of normalised features (facts) to be passed into a range of classification models such as, k-nearest neighbours, support vector machine, decision tree, random forest, adaptive boosting and gradient boosting classifiers. These results will help with the implementation of an ensemble model and the possible inclusion of deep learning neural networks, the most likely candidate being an LST Recurrent Neural Network. Another approach would be to use an ensemble of SVM and a boosting classifier which might help to decrease errors and enhance generalisability. 

A quarter has about 60 days. In my analysis I want to incorporate information, i.e. features, that occurs in a quarter to make prediction regarding the earnings estimate at the end of that quarter. The first bundle of features transforms down to a single number for each earnings estimate. I will also incorporate features that develop sequentially and use machine learning to identify patterns that form from 5, 10 and 20 days before the announcement. I believe that such an analysis will show some leakage of inside information and that the public data available will be enough for the learning models to classify the likelihood of an earnings surprise.  With this type of deep analysis there is the potential of a lot of statistical information being locked up in relationships of the features, which might not at first be apparent to human analysts as compared to that of learning algorithms. There are important traditional models of time series modelling that we would incorporate as features to identify the higher dimensional information as computed by the machine learning models. These statistical models include moving average, exponential smoothing averages and other.

Machine learning has been used in finance in a number of studies. Bagheri et al. (2014) used an adaptive networked based fuzzy inference system to forecast financial time series for currencies. Hu et al. (2015) used a hybrid evolutionary trend following algorithms to introduce a trading algorithm which selects stocks based on different indicators.  De Oliveira et al. (2013) use economic and financial theory in combination with fundamental and technical and time series analysis to predict price behaviour with the use of artificial neural networks (ANN). Patel et al. (2015) compared multiple prediction models such as ANN, SVM and random forest. Booth et al. (2014) propose an expert system based on machine learning techniques to predict the return on seasonal events and develop a profitable trading strategy. The above mentioned studies inspired many of the benchmark models used in this study. I will also incorporate novel feature selection procedures and ensemble models to identify not only the baseline, but also the best accuracy achievable for the task at hand. I will report on the results of all the models to prohibit data snooping.

The majority of research in this field focused on the price movements of stocks, indexes and currencies. These studies are also limited in sample size, most opting to analyse a handful number of stocks using daily data. It is of my view that it is difficult to rule out random results if the database is not wide enough. Very few studies look at financial outcomes, furthermore, no study has as yet used modern machine learning techniques to investigate the probability of financial events. This study can produce a novel system to help investors and market makers managing their stock ownership before earnings announcements, for not just profit maximisation, but also risk management purposes. Trading system strategies come from a large range of fields, be it econometric modelling, evolutionary computation, news mining or machine learning. In the 1960s trading rules based on technical indicators were said not to be profitable (Fama, Blume, 1966). However, these indicators were never specifically applied to event prediction. Xiao et al (2013) demonstrated the power of ensembles in financial market forecasting. They show that the flexibility of the ensemble approach is key to their ability to capture complex nonlinear relationships. All the studies that used ensembles demonstrated the ability to avoid overfitting compared to using standalone system. This study will also make use of ensemble techniques. 
Hypothesis: 

-	There is hidden relationships in technical information that can predict earnings surprises.
-	The inclusion of manually engineered features will significantly improve model accuracy in predicting financial events.
-	The combination of financial and technical elements will significantly increase the accuracy of predicting earnings surprises.
-	The addition of twitter and newspaper NLP analysis will further strengthen the model accuracy (possibly a separate study).
-	For financial prediction a more comprehensive model with more features and data will outperform a smaller dataset with less features.
-	An ensemble of models would outperform single and benchmark models.
-	A final neural networks model that uses traditional machine learning outputs as its inputs will outperform an ensemble of classifiers mean.
-	Hyperparameter optimisation of the learning rate, the number of estimators for boosting, the number of nodes will lead to better accuracy overall.
-	Smaller firm accuracy would be higher than large firm accuracy as a result of being less covered/scrutinised and more informationally inefficient.

Pre-thesis results: 

-	Benchmark: using the test data, the analyst estimate is 5% below the actual earnings 43% of the time. 
-	The trained model predicts a 60%+ accuracy as to whether the analyst’s estimate would be 5% below the actual earnings.  
-	I trained the model on beating the estimate by 5% to ensure that there would be a stock price reaction.
-	Therefore, with the preliminary model, using 5 hand engineered and two time relevant predictors, the model can predict with more than 60% accuracy whether the analyst’s estimates will be beaten by the actuals with more than 5%, i.e. predict an earnings surprise. 
-	The attached results have not been tested on a holdout set. It is of my believe that the accuracy would fall between a range of 55%-60% once this is done. However, this value can also improve with good feature and model selection procedures. 




---



```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import talib as ta
from glob import glob
import pdb
from ipykernel import kernelapp as app
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import itertools as it

#from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn

from __future__ import division
import random 
def warn(*args, **kwargs): pass

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

```


```python
seed = 7
np.random.seed(seed)
```


```python
ann_cusip = pd.read_csv("ann_cusip_bo.csv", dtype=str)
ann_cusip["CUSIP"]= ann_cusip["CUSIP"].map(lambda x: x[:6] + '10')
ann_tic = pd.read_csv("ann_tic_bo.csv", dtype=str)

ann_cusip = pd.read_csv("ann_cusip_bo.csv", dtype=str)
ann_cusip["CUSIP"]= ann_cusip["CUSIP"].map(lambda x: x[:6] + '10')
ann_tic = pd.read_csv("ann_tic_bo.csv", dtype=str)


connect = pd.merge(ann_cusip, ann_tic, left_on=["ANNDATS_ACT", "OFTIC"], right_on=["anndats", "OFTIC"], how="outer")

#connect.drop("STATPERS",1, inplace=True)
connect.drop_duplicates(inplace=True)
connect.dropna(inplace=True)

prices = pd.read_csv("prices_bo.csv", dtype=str)

prices["CUSIP"]= prices["CUSIP"].map(lambda x: x[:6] + '10')



fiscal = pd.read_csv("fiscal_bo.csv", dtype=str)
fiscal.dropna(inplace=True)
fiscal["cusip"] = fiscal["cusip"].dropna()

fiscal["cusip"]= fiscal["cusip"].map(lambda x: x[:6] + '10')


```


```python

merged_2 = pd.merge(prices, fiscal, left_on=["date", "CUSIP"], right_on=["rdq", "cusip"], how="outer")
merged_2.to_csv("merged_2_bo.csv")
merged_3 = pd.merge(merged_2, connect, left_on=["date", "CUSIP"], right_on=["anndats","CUSIP"], how="outer")
merged_3.to_csv("merged_3_bo.csv")
merged = merged_3.fillna(method="backfill")
merged = merged.fillna(method="ffill")
merged.to_csv("merged_bo.csv")

merged.rename(columns={'BID':'close', 'ASKHI':'high', 'BIDLO':'low', 'OPENPRC':'open', 'surpmean':'estimate', 'datadate_x':'datadate', 'tic_x':'ticker', 'gvkey_x':'gvkey', 'VOL':'volume','fyearq':'year','fqtr':'qtr'}, inplace=True)

# I have included anndats, previously excluded. 
merged = merged[["cusip", "anndats", "open", "high", "low", "close", "volume", "estimate", "actual", "datadate", "year","qtr"]]
```


```python
r = .9523
#.968 

merged["beat"] = np.where((merged["actual"].map(lambda x: float(x)))>(merged["estimate"].map(lambda x: float(x)/r)),1,0)

merged = merged.drop(merged[merged.anndats < merged.datadate].index)

merged = merged.drop(merged[(merged.anndats.astype(int) - merged.datadate.astype(int))>11100].index)

# df = df.drop(df[df.score < 50].index)

merged.set_index(['year', 'qtr', 'cusip' ], inplace=True)

merged.drop("ticker", inplace=True)

merged.to_csv("merged_complete.csv")

```


```python

```


```python
# This establishes the model framework. 

def target_f(merged, cusi):
    saved = pd.DataFrame()
    new_group = merged.xs(cusi, level='cusip')
    model_frame = new_group["beat"].groupby(level=['year', 'qtr']).agg([np.median]).reset_index()
    bruse = model_frame
    model_frame.rename(columns={"median":"target"}, inplace=True)
    model_frame = model_frame["target"].astype(int)
    saved["target_"+ cusi] = model_frame
    saved["year"+ cusi] = bruse["year"]
    saved["qtr"+ cusi] = bruse["qtr"]
    
    return saved 


def mom50_f(merged, cusi): 
    
    saved = pd.DataFrame()
    d = pd.DataFrame()
    new_group = merged.xs(cusi, level='cusip')
    d["close"] = new_group["close"]
    d.reset_index()

    momentums = pd.DataFrame()
    ticks = pd.DataFrame()
    ticks = d["close"].dropna().as_matrix()
    ticks = np.array(ticks,dtype='f8')
    mom1 = ta.MOM(ticks, 10)
    mom2 = np.where(mom1 > 0, 1, 0)
    d["mom"] = mom2

    market_df = d["mom"].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    cnt = np.array(market_df['len']).astype(np.float64)
    sm  = np.array(market_df['sum']).astype(np.float64)
    here= np.where(np.divide(sm,cnt) > 0.5, 1, 0)
    saved["mom50_"+ cusi] = here
    saved["year"+ cusi] = market_df["year"]
    saved["qtr"+ cusi] = market_df["qtr"]
    return saved 
    

def ma20_f(merged, cusi):
    saved = pd.DataFrame()
    d = pd.DataFrame()
    new_group = merged.xs(cusi, level='cusip')
    d["close"] = new_group["close"]
    #d.reset_index()
    
    ticks = d["close"].dropna().as_matrix()
    ticks = np.array(ticks,dtype='f8')
    mom1 = ta.MA(ticks, 20)
    df = pd.DataFrame({'mom1':mom1.tolist()})
    df = df.fillna(method="bfill")

    de = pd.DataFrame({'close':ticks.tolist()})

    # NB with the moving averages, you would automatically have those first days cut off.
    # You can backfill it, period long enough for that not to be an issue.  

    values = (de['close'] - df['mom1'])
    mom2 = pd.DataFrame()
    d['price_above'] = np.where(values > 0, 1, 0)

    #market_df = mom2['price_above'].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    market_df = d['price_above'].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    cnt = np.array(market_df['len']).astype(np.float64)
    sm  = np.array(market_df['sum']).astype(np.float64)
    here= np.where(np.divide(sm,cnt) > 0.5, 1, 0)
    saved["ma20_"+ cusi] = here
    saved["year"+ cusi] = market_df["year"]
    saved["qtr"+ cusi] = market_df["qtr"]

    return saved

   
def day_f(merged, cusi): 

    saved = pd.DataFrame()
    cusip = merged.index.levels[2].unique()

    new_group = merged.xs(cusi, level='cusip')
    d = new_group


    d['upday'] = np.where(new_group['close'].convert_objects(convert_numeric=True) - new_group['open'].convert_objects(convert_numeric=True) > 0, 1, 0)


    market_df = d["upday"].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    cnt = np.array(market_df['len']).astype(np.float64)
    sm  = np.array(market_df['sum']).astype(np.float64)
    here= np.where(np.divide(sm,cnt) > 0.5, 1, 0)
    saved["day_"+ cusi] = here
    saved["year"+ cusi] = market_df["year"]
    saved["qtr"+ cusi] = market_df["qtr"]
    
    return saved



def ema10_f(merged, cusi):
    saved = pd.DataFrame()
    d = pd.DataFrame()
    new_group = merged.xs(cusi, level='cusip')
    d["close"] = new_group["close"]
    #d.reset_index()

    
    ticks = d["close"].dropna().as_matrix()
    ticks = np.array(ticks,dtype='f8')
    mom1 = ta.EMA(ticks, 10)
    df = pd.DataFrame({'mom1':mom1.tolist()})
    df = df.fillna(method="bfill")

    de = pd.DataFrame({'close':ticks.tolist()})

    # NB with the moving averages, you would automatically have those first days cut off.
    # You can just backfill it, there is noissue in that. 

    values = (de['close'] - df['mom1'])
    mom2 = pd.DataFrame()
    d['price_above'] = np.where(values > 0, 1, 0)

    #market_df = mom2['price_above'].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    market_df = d['price_above'].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    cnt = np.array(market_df['len']).astype(np.float64)
    sm  = np.array(market_df['sum']).astype(np.float64)
    here= np.where(np.divide(sm,cnt) > 0.5, 1, 0)
    saved["ema10_"+ cusi] = here
    saved["year"+ cusi] = market_df["year"]
    saved["qtr"+ cusi] = market_df["qtr"]

    return saved



def mom_vol_f(merged, cusi): 
    
    saved = pd.DataFrame()
    d = pd.DataFrame()
    new_group = merged.xs(cusi, level='cusip')
    d["volume"] = new_group["volume"]
    d.reset_index()

    momentums = pd.DataFrame()
    
    ticks = d["volume"].dropna().as_matrix()
    ticks = np.array(ticks,dtype='f8')
    mom1 = ta.MOM(ticks, 10)
    mom2 = np.where(mom1 > 0, 1, 0)
    d["v_mom"] = mom2

    market_df = d["v_mom"].groupby(level=['year', 'qtr']).agg([np.mean, np.sum, np.std, len]).reset_index()

    cnt = np.array(market_df['len']).astype(np.float64)
    sm  = np.array(market_df['sum']).astype(np.float64)
    here= np.where(np.divide(sm,cnt) > 0.5, 1, 0)
    saved["mom_vol_"+ cusi] = here
    saved["year"+ cusi] = market_df["year"]
    saved["qtr"+ cusi] = market_df["qtr"]
    
    return saved 
    

```


```python
# Aggregation of features

saved = pd.DataFrame()
cusip = merged.index.levels[2].unique()

day_cusip = pd.DataFrame()
day = pd.DataFrame()

ma20_cusip = pd.DataFrame()
ma20 = pd.DataFrame()

mom50_cusip = pd.DataFrame()
mom50 = pd.DataFrame()

ema10_cusip = pd.DataFrame()
ema10 = pd.DataFrame()

mom_vol_cusip = pd.DataFrame()
mom_vol = pd.DataFrame()

target_cusip = pd.DataFrame()
target = pd.DataFrame()


for cusi in cusip:
    day_cusip = day_f(merged, cusi)
    day = pd.concat((day, day_cusip),axis=1) 
   
    ma20_cusip = ma20_f(merged, cusi)
    ma20 = pd.concat((ma20,ma20_cusip), axis=1)
    
    mom50_cusip = mom50_f(merged, cusi)
    mom50 = pd.concat((mom50,mom50_cusip),axis=1)
    
    ema10_cusip = ema10_f(merged, cusi)
    ema10 = pd.concat((ema10,ema10_cusip), axis=1)
    
    mom_vol_cusip = mom_vol_f(merged, cusi)
    mom_vol = pd.concat((mom_vol ,mom_vol_cusip),axis=1)
    
    target_cusip = target_f(merged, cusi)
    target = pd.concat((target, target_cusip), axis=1)
    
    
varlist = {"target_":target,
           "day_":day,
           "ma20_":ma20,
           "mom50_":mom50,
           "ema10_":ema10,
           "mom_vol_":mom_vol}

```

    /Users/dereksnow/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:89: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
## All of this create the time dummies data. 

keys = [c for c in day if c.startswith("year")]
frame_1 = pd.melt(day, value_vars=keys, value_name="year")

keys = [c for c in day if c.startswith("qtr")]
frame_2 = pd.melt(day, value_vars=keys, value_name="qtr")

frame = pd.concat((frame_1,frame_2), axis=1)
avg = frame 

cols_to_transform = ['year', 'qtr']
frame = pd.get_dummies(avg, columns = cols_to_transform)
frame = pd.concat((frame, avg), axis=1)

frame["myver"] = frame.ix[:,[1]]
frame = frame.drop(["variable"], axis=1)

```


```python
frame.to_csv("framecsv.csv")
```


```python
frame_full = pd.DataFrame()

for i, v in varlist.iteritems():
    keys = [c for c in v if c.startswith(i)]
    frame_a = pd.melt(v, value_vars=keys, value_name=i)
    frame_full = pd.concat((frame_full,frame_a), axis=1)
 

frame_full_1 = frame_full.drop(["variable"], axis=1)
fire = frame_full_1


```


```python
fire.to_csv("fire3444.csv")
```


```python
fire_1 = pd.concat((fire, frame), axis=1)

"""
fire.to_csv("fire.csv")

frame.to_csv("frame.csv")
"""

frame_full = fire_1
frame_full = frame_full.drop_duplicates()
frame_full = frame_full.dropna()

frame_full = frame_full.dropna()
frame_full.reset_index(inplace=True, drop=True)

#frame_full.reset_index(inplace=True, drop=True)
#frame_full = frame_full.drop_duplicates()

##########################

frame_full["target_p"] = frame_full["target_"].shift(-1)
frame_full["target_p2"] = frame_full["target_"].shift(-2)

X_first_1 = frame_full 
X_first_1 = X_first_1.drop(["myver", "year", "qtr"],axis=1)
X_first = X_first_1.dropna(axis=0)

#fire_1.to_csv("fire_1.csv")
#frame_full.to_csv("frame_full.csv")


##########################

# -------------------- Here is where I create all the alternative targets. 

bloom = frame_full[["year", "qtr", "myver", "target_"]]

bloom_1 = bloom.set_index(['year', 'qtr'])

here = bloom_1.groupby(level=['year', 'qtr']).mean()

here_1 = here.reset_index()

here_1.rename(columns={'target_': 'ind_target'}, inplace=True)

framed = pd.merge(frame_full, here_1, on=["year","qtr"], how="outer")

framed.to_csv("framed.csv")

#framed.rename(columns={'target_': 'ind_target'}, inplace=True)



```


```python
X_first.to_csv("X_first.csv")
```


```python
# This feauture was not contributary. 

# frame_full["target_x2"] = frame_full["target_"].shift(-2) if frame_full["target_"].shift(-1) == frame_full["target_"].shift(-2)
# frame_full["target_x2"] = np.where(np.logical_and(frame_full["target_"].shift(-3) == frame_full["target_"].shift(-2), frame_full["target_"].shift(-2) == 1), 1,0)
# This one actually killed me, its addition led to worse results


framed_1 = framed.dropna(axis=0)

framed_1.reset_index(inplace=True, drop=True)

framed_1.drop(["myver", "year", "qtr"],axis=1,inplace=True)


```

    /Users/dereksnow/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
framed_1.to_csv("framed.csv")
```


```python
X = X_first.drop(["target_"], axis=1)

# X_first performs slightly better than framed_1  

# Two alternatives, X_first of framed_1

# ind_target. 

y = X_first["target_"]
```


```python
### For data transfer
#y = y.to_frame()
### y should be series if not trnasfering 
```


```python
#X.to_csv("X.csv")
#y.to_csv("y.csv")
```


```python

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=43)

# 0.33, 43

this = sum(y_test)/len(y_test)
```


```python
this
#bench
```




    0.43787672564650981




```python
classifiers = [
    KNeighborsClassifier(4),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()]



log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
from sklearn.cross_validation import cross_val_score
framed = np.array

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    y_predict = clf.predict(X_test)
    print metrics.accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    
    print("Accuracy: {:.4%}".format(acc))
    
    #scores = cross_val_score(clf, X , y , cv=5)
    #print scores
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
    
    
    ###
    #print y_predict
    #print y_test.as_matrix()
    
    y_predict = clf.predict_proba(X_test)
    ll = log_loss(y_test, y_predict)
    print("Log Loss: {}".format(ll))
    
    #print y_predict
    #print y_test.as_matrix()
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
    # However at this step we still haven't looped and walked through the different KNNs.
    #scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    # Here we folded 10 times, this has nothing to do with the previous cell, we have not incorporated the 5 split]
    
print("="*30)

# Cross val is not actually neede you dont need that extra 20%, already have enough data. 


```

    ==============================
    KNeighborsClassifier
    ****Results****
    0.574761812172
    Accuracy: 57.4762%
    Log Loss: 3.14670788089
    ==============================
    SVC
    ****Results****
    0.626288158662
    Accuracy: 62.6288%
    Log Loss: 0.641448022373
    ==============================
    DecisionTreeClassifier
    ****Results****
    0.596733424072
    Accuracy: 59.6733%
    Log Loss: 3.88603377725
    ==============================
    RandomForestClassifier
    ****Results****
    0.594205716508
    Accuracy: 59.4206%
    Log Loss: 1.40215773532
    ==============================
    AdaBoostClassifier
    ****Results****
    0.640676647871
    Accuracy: 64.0677%
    Log Loss: 0.691001912187
    ==============================
    GradientBoostingClassifier
    ****Results****
    0.640871086914
    Accuracy: 64.0871%
    Log Loss: 0.637834495096
    ==============================
    GaussianNB
    ****Results****
    0.621816060665
    Accuracy: 62.1816%
    Log Loss: 0.769997339871
    ==============================



```python
k_range = range(1,26)
# I think range just instantiates a vector variable
# Below is also an instantiation of a vector variable
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print scores
```

    [0.57353184449958639, 0.56658395368072789, 0.60397022332506201, 0.59851116625310175, 0.62580645161290327, 0.6165425971877585, 0.63225806451612898, 0.62861869313482222, 0.6387096774193548, 0.63291976840363939, 0.63622828784119112, 0.64251447477253931, 0.64267990074441683, 0.64168734491315138, 0.64582299421009104, 0.64267990074441683, 0.64946236559139781, 0.64681555004135649, 0.64747725392886679, 0.65144747725392882, 0.64863523573200987, 0.64698097601323412, 0.6512820512820513, 0.64747725392886679, 0.64946236559139781]



```python
%who str
```


```python

```
