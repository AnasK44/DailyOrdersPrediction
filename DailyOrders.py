import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb 
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform,randint
from xgboost import plot_importance


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


path = r'C:\Users\LENOVO\Desktop\res2.csv';

ori = pd.read_csv(path);
print(ori.head())
ori = pd.DataFrame(ori)
ori['Order Date'] = pd.to_datetime(ori['Order Date'] ,dayfirst=True)
ori['Order Date'] = ori['Order Date'].dt.date


print(ori.head())
print(f"Total number of orders in dataset: {ori['Order ID'].nunique()}")
print(ori.shape)




ori.columns = ['Order_ID', 'Date', 'item', 'quantity', 'price', 'total_products_in_cart']
print(ori.head())

from pandasql import sqldf
pysqldf = lambda q : sqldf(q,globals());

def load_q(path):
    with open(path) as file:
        return file.read();
    

pathq = r'C:\Users\LENOVO\Desktop\ques\t2Daily.sql'    
query = load_q(pathq);
DailyOrders = pysqldf(query);
DailyOrders = pd.DataFrame(DailyOrders);

print(DailyOrders.head());


plt.rcParams.update({'figure.figsize': (17, 3), 'figure.dpi':300})
fig, ax = plt.subplots()
sns.lineplot(data=DailyOrders.tail(110), x='date', y='total_number')
plt.grid(linestyle='-', linewidth=0.3)
ax.tick_params(axis='x', rotation=90)



df = pd.DataFrame(DailyOrders)

# Convert 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Set 'date' column as the index
df.set_index('date', inplace=True)

# Reindex with a complete date range and fill missing values with zero
complete_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(complete_date_range)


# Plot the chart
df.plot(y='total_number', linestyle='solid', figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('numer of orders')
plt.title('Time line for orders count')
plt.show()

DailyOrders = df.reset_index(inplace = False)

DailyOrders.rename(columns={'index': 'date'}, inplace=True)

#filling missing values with zero
DailyOrders['total_number'].fillna(0, inplace=True);
#Clearly there is one outlier in 2019 somewhere, and also there are a lot of data points missing
#Getting the value of yesteday & filtering data because of there're some missing values
# we are going to get from greater than aug of 2016
DailyOrders['yesterday_number'] = DailyOrders['total_number'].shift(1)
DailyOrders['date'] = pd.to_datetime(DailyOrders['date'] ,dayfirst=True)
DailyOrders = DailyOrders.loc[DailyOrders['date'] > '2016-08-01']
DailyOrders['date'] = DailyOrders['date'].dt.date

print(DailyOrders.head())


######## create a new column with the date value of the same day last week
from datetime import timedelta
DailyOrders['last_week_date'] = DailyOrders['date'] - timedelta(days=7)
DailyOrders['last_week_date'] = pd.to_datetime(DailyOrders['last_week_date'] ,dayfirst=True)
DailyOrders['last_week_date'] = DailyOrders['last_week_date'].dt.date
DailyOrders['last_week_value'] = DailyOrders.apply(lambda row: DailyOrders.loc[DailyOrders['date'] == row['last_week_date'], 'total_number'].iloc[0] 
                                                   if len(DailyOrders.loc[DailyOrders['date'] == row['last_week_date'], 'total_number']) > 0 else 0, axis=1)
print(DailyOrders.head())

#getting the sum of the last week 
DailyOrders['last_week_sum'] = DailyOrders.apply(lambda row: DailyOrders.loc[(DailyOrders['date'] >= row['last_week_date']) & (DailyOrders['date'] < row['date']), 'total_number'].sum(), axis=1)


DailyOrders['date'] = pd.to_datetime(DailyOrders['date'] ,dayfirst=True)
print(DailyOrders.head())

#Set index
DailyOrders = DailyOrders.set_index('date');
DailyOrders.index = pd.to_datetime(DailyOrders.index)

#Spliting Data for Train and test 
train = DailyOrders.loc[DailyOrders.index <'2019-07-01']
test = DailyOrders.loc[DailyOrders.index >= '2019-07-01']




def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek.astype(int)
    df['month'] = df.index.month.astype(int)
    df['year'] = df.index.year.astype(int)
    df['dayofyear'] = df.index.dayofyear.astype(int)
    df['dayofmonth'] = df.index.day.astype(int)
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    return df

df = create_features(DailyOrders)
train = create_features(train)
test = create_features(test)

print(df.head())


FEATURES = ['yesterday_number','last_week_sum', 'last_week_value','dayofyear', 'weekofyear', 'dayofweek', 'month','year','dayofmonth']
TARGET = 'total_number'


X_old = df[FEATURES]
y_old = df[TARGET]



#############################LET's go inital model#############################
reg = xgb.XGBRegressor()

reg.fit(X_old, y_old , eval_set=[(X_old, y_old)])


# get feature importance and ranking
important_values = reg.feature_importances_
sorted_idx = np.argsort(important_values)[::-1]
print(important_values)

# Plot feature importance
xgb.plot_importance(reg)

# remove features with importance = 0
important_var_gain = [(X_old.columns[index], important_values[index]) for index in sorted_idx if important_values[index] > 0]
# we have 400 features with importance value>0
len(important_var_gain)

# restructure data and refit
important_varlist = [it[0] for it in important_var_gain]
print(important_varlist)

################################Splitting #####################################

X_train = train[important_varlist]
y_train = train[TARGET]

X_test = test[important_varlist]
y_test = test[TARGET]



###Plotting 
fig, ax = plt.subplots(figsize=(15, 5))
pd.DataFrame(train[train.index >= '2019-01-01']).plot( y='total_number',ax=ax,label='Training Set', title='Data Train/Test Split',)
pd.DataFrame(test).plot( y='total_number',ax=ax ,label='Test Set')
ax.axvline('2019-07-01', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()       

print(test.shape)


########################### RandomizedSearchCV ################################
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform,randint

xgbreg = xgb.XGBRegressor(objective = 'reg:squarederror')
param_dist = {'n_estimators': randint(1000, 3000),
              'learning_rate': uniform(0.01, 0.06),
              'subsample': [0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              'gamma' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
              'colsample_bytree': [0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
              'min_child_weight': [1, 2, 3, 5, 7],
              'reg_alpha' : [0,1,2,3,4,5,6,7,8,9,10],
              'reg_lambda' : [0,1,2,3,4,5,6,7,8,9,10]
             }

# verbose: integer
# Controls the verbosity: the higher, the more messages.

regcv = RandomizedSearchCV(xgbreg, param_distributions = param_dist, 
    n_iter = 25, scoring = 'r2', cv = 3,
    error_score = 0, verbose = 10, n_jobs = -1)

search = regcv.fit(X_train, y_train)

search.best_params_

######################### Actual Model ########################################
reg = xgb.XGBRegressor(objective = 'reg:squarederror',
                  
                      learning_rate=0.010286668188327738,
                      gamma = 0 ,
                      reg_alpha= 10,  
                      reg_lambda= 4,
                      colsample_bytree= 0.85,
                      max_depth =  7,
                      min_child_weight = 1,
                      n_estimators= 1078,
                      booster='gbtree',
                      subsample = 0.7
 )

reg.fit(X_train, y_train, early_stopping_rounds=100, 
        eval_metric="mae", eval_set=[(X_test, y_test)])



y_pred = reg.predict(X_test)

################## R2 score ##############################
#Calculate the R² score of the predictions on the testing data
r2_score = reg.score(X_test, y_test)
#Convert R² to percentage accuracy
percentage_accuracy = r2_score * 100
print("Percentage accuracy: {:.2f}%".format(percentage_accuracy))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred)*100)
## 82%

########### MSE ##############
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

################ RMSE ############
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae) ## 20
print("RMSE" , np.sqrt(mse))

########## MEAN #############
accuracy = 100 * (1 - mae/y_pred.mean())
print(f"The accuracy  MEAN of the XGBoost model is {accuracy:.2f}%");



################## PLOTTING predicted vs actual #####################
test.reset_index(inplace = True)

new_df = test.copy();


y_pred = pd.Series(y_pred)
new_df['predicted'] = y_pred.values

fig, ax = plt.subplots(figsize=(10, 5))
pd.DataFrame(test).plot(x='date', y = 'total_number',ax=ax,label='Real Data', title='Predicted Data/Real Data')
pd.DataFrame(new_df).plot(x='date', y = 'predicted',ax=ax ,label='Predicted')
ax.legend(['Real Data', 'Predicted Data'])
plt.show()   



import pickle
with open('orders.pkl', 'wb') as f:
    pickle.dump(reg, f)



