# -*- coding: utf-8 -*-
"""laptop_price_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U4TQE3ZvhvXj9GfGScWnv82Ya6u_HPKj
"""

import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('/content/laptop.csv')
df.head()

df.drop(columns=["Unnamed: 0.1","Unnamed: 0"],inplace=True)

df.head()

df['Price'].unique()

df['Ram']=df['Ram'].str.replace('GB','')
df['Weight']=df['Weight'].str.replace('kg','')

df.head()

df.isnull().sum()

df.info()

df['Company'].mode()[0]
df['Company'].fillna(df['Company'].mode()[0],inplace=True)

df['TypeName'].mode()[0]
df['TypeName'].fillna(df['TypeName'].mode()[0],inplace=True)

df['Inches'].mode()[0]
df['Inches'].fillna(df['Inches'].mode()[0],inplace=True)

df['Inches']=df['Inches'].replace('?','15.6')

df['Inches']=df['Inches'].astype('float')

df['ScreenResolution'].mode()[0]
df['ScreenResolution'].fillna(df['ScreenResolution'].mode()[0],inplace=True)

df['Cpu'].mode()[0]
df['Cpu'].fillna(df['Cpu'].mode()[0],inplace=True)

df['Ram'].mode()[0]

df['Ram']=df['Ram'].replace('nan','8')

df['Ram'].fillna(df['Ram'].mode()[0],inplace=True)

df['Ram'].unique()

df['Ram']=df['Ram'].astype('float64')

df['Memory'].mode()[0]

df['Memory']=df['Memory'].replace('?','256GB SSD')

df['Memory'].fillna(df['Memory'].mode()[0],inplace=True)

df['Gpu'].mode()[0]
df['Gpu'].fillna(df['Gpu'].mode()[0],inplace=True)

df['OpSys'].mode()[0]

df['OpSys']=df['OpSys'].replace('nan','Windows 10')

df['OpSys'].fillna(df['OpSys'].mode()[0],inplace=True)

df['Weight'].mode()[0]

df['Weight']=df['Weight'].replace('?','2.2')

df['Weight'].fillna(df['Weight'].mode()[0],inplace=True)

df['Weight']=df['Weight'].astype('float64')

sns.distplot(df['Price'])

df['Company']=df['Company'].str.strip()
df['TypeName']=df['TypeName'].str.strip()
df['ScreenResolution']=df['ScreenResolution'].str.strip()
df['Cpu']=df['Cpu'].str.strip()
df['Memory']=df['Memory'].str.strip()
df['Gpu']=df['Gpu'].str.strip()
df['OpSys']=df['OpSys'].str.strip()

df['Price'].median()

df['Price'].unique()

df['Price']=df['Price'].replace('nan',52161.12)

df['Price'].fillna(df['Price'].median(),inplace=True)

df.isnull().sum()

df.info()

sns.boxplot(df['Inches'])
px.box(df,y='Inches')

df.describe()['Inches']

q1=df.describe()["Inches"]["25%"]

q3=df.describe()["Inches"]["75%"]

IQR=q3-q1

upper_limit=q3+1.5*IQR

lower_limit=q1-1.5*IQR

df["Inches"]=df["Inches"].clip(lower_limit,upper_limit)

px.box(df,y='Inches')

px.box(df,y='Ram')

q1=df.describe()["Ram"]["25%"]

q3=df.describe()["Ram"]["75%"]

IQR=q3-q1

upper_limit=q3+1.5*IQR

lower_limit=q1-1.5*IQR

df["Ram"]=df["Ram"].clip(lower_limit,upper_limit)

px.box(df,y='Ram')

px.box(df,y='Weight')

q1=df.describe()["Weight"]["25%"]

q3=df.describe()["Weight"]["75%"]

IQR=q3-q1

upper_limit=q3+1.5*IQR

lower_limit=q1-1.5*IQR

df["Weight"]=df["Weight"].clip(lower_limit,upper_limit)

px.box(df,y='Weight')

df['Price']

df['Price']=df['Price'].apply(lambda x: float("{:.2f}".format(x)))

df.head()["Price"]

px.box(df,y='Price')

q1=df.describe()["Price"]["25%"]

q3=df.describe()["Price"]["75%"]

IQR=q3-q1

upper_limit=q3+1.5*IQR

lower_limit=q1-1.5*IQR

df["Price"]=df["Price"].clip(lower_limit,upper_limit)

px.box(df,y='Price')

sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

sns.scatterplot(x=df['Inches'],y=df['Price'])

df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

df.sample(5)

df['Touchscreen'].value_counts().plot(kind='bar',color=['red','green'])

sns.barplot(x=df['Touchscreen'],y=df['Price'],hue=df['TypeName'])

sns.barplot(x=df['Touchscreen'],y=df['Price'],hue=df['Company'])

df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

df.sample(5)

df['Ips'].value_counts().plot(kind='bar',color=['red','green'])

sns.barplot(x=df['Ips'],y=df['Price'],hue=df['TypeName'])

df['ScreenResolution'].unique()

df['ScreenResolution'].str.split('x').str[-1]

df['ScreenResolution'].str.split('x',n=1,expand=True)

newSR=df['ScreenResolution'].str.split('x',n=1,expand=True)

df['X_res']=newSR[0]

df['Y_res']=newSR[1]

df.sample(5)

df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

df.head()

df.info()

df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')

df.info()

df['ppi']=(((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']).astype('float')

df.head()

df.drop(columns=['ScreenResolution'],inplace=True)

df.head()

df['Cpu'].unique()
# making five category of Cpu variants 1:i7 2:i5 3:i3 4:Intel pantium and Intel Celeron others intel 5: AMD

df['Cpu_bnd']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

df.head()

def fetch_processor(text):
    if text=='Intel Core i7' or text=='Intel Core i5' or text=='Intel Core i3':
        return text
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

df['Cpu_brnd']=df['Cpu_bnd'].apply(fetch_processor)

df.head()

df.sample(5)

df['Ram']=df['Ram'].astype('int')
df.sample(5)

df['Ram'].value_counts()

sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

df['Ram'].value_counts().plot(kind='bar')

df['Cpu_brnd'].value_counts().plot(kind='bar')

sns.barplot(x=df['Cpu_brnd'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

df['Memory'].unique()
#4 section of memory SSD,HDD,Flash and Hybrid

df['Memory']=df['Memory'].astype(str).replace('\.0', '', regex=True)
df.sample(5)

df['Memory']=df['Memory'].str.replace('GB','')

df.head()

df['Memory']=df['Memory'].str.replace('TB','000')

df.sample(5)

df.drop(columns=['Cpu_bnd'],inplace=True)

df.sample(5)

new=df['Memory'].str.split('+',n=1,expand=True)

df['first']=new[0]

df.head()

df['first']=df['first'].str.strip()

df['Second']=new[1]

df.head()

df['Layer1HDD']=df['first'].apply(lambda x:1 if 'HDD' in x else 0)

df['Layer1SSD']=df['first'].apply(lambda x:1 if 'SSD' in x else 0)

df['Layer1Hybrid']=df['first'].apply(lambda x:1 if 'Hybrid' in x else 0)

df['Layer1Flash_Storage']=df['first'].apply(lambda x:1 if 'Flash Storage' in x else 0)

df.sample(5)

df['first']=df['first'].str.replace(r'\D','')

df.head()

df['Second'].fillna('0',inplace=True)

df.head()

df['Layer2HDD']=df['Second'].apply(lambda x:1 if 'HDD' in x else 0)

df.head()

df['Layer2SSD']=df['Second'].apply(lambda x:1 if 'SSD' in x else 0)

df['Layer2Hybrid']=df['Second'].apply(lambda x:1 if 'Hybrid' in x else 0)

df['Layer2Flash_Storage']=df['Second'].apply(lambda x:1 if 'Flash Storage' in x else 0)

df.head()

df['Second']=df['Second'].str.replace(r'\D','')

df.head()['first']

df['first']=df['first'].str.split(' ',n=1,expand=True)[0]

df['first'].unique()

df.head()

df['first']=df['first'].astype(int)

df['Second'].unique()

df['Second']=df['Second'].str.strip(' ')

df['Second']=df['Second'].str.split(' ',n=1,expand=True)[0]

df['Second']=df['Second'].astype(int)

df['HDD']=(df['first']*df['Layer1HDD']+df['Second']*df['Layer2HDD'])

df['SSD']=(df['first']*df['Layer1SSD']+df['Second']*df['Layer2SSD'])

df['Hybrid']=(df['first']*df['Layer1Hybrid']+df['Second']*df['Layer2Hybrid'])

df['Flash_Storage']=(df['first']*df['Layer1Flash_Storage']+df['Second']*df['Layer2Flash_Storage'])

df.head(10)

df.drop(columns=['first','Second','Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'],inplace=True)

df.sample(5)

df['Gpu'].unique()
# extract the brand name wise

df['Gpu_brnd']=df['Gpu'].apply(lambda x:x.split()[0])

df.head()

df['Gpu_brnd'].value_counts()

sns.barplot(x=df['Gpu_brnd'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

df.drop(columns=['Gpu'],inplace=True)

df.head()

df['OpSys'].unique()

df['OpSys'].value_counts()

sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

def os_cat(inp):
    if inp=='Windows 10' or inp=='Windows 7' or inp=='Windows 10 S':
        return 'Windows'
    elif inp== 'macos' or inp== 'Mac OS X':
      return 'Mac'
    else:
      return 'Others/No OS/Linux'

df['OS_typ']=df['OpSys'].apply(os_cat)

df.sample(5)

df.drop(columns=['OpSys'],inplace=True)

sns.barplot(x=df['OS_typ'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

sns.distplot(df['Weight'])

sns.distplot(np.log(df['Price']))

df.head()

df.drop(columns=['Cpu','Memory','X_res','Y_res','Inches'],inplace=True)

df.head()

x=df.drop(columns=['Price'])
y=np.log(df['Price'])

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

X_train

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

df.head(1)

"""Linear Regression"""

categorical_columns=["Company","TypeName","Cpu_brnd","Gpu_brnd","OS_typ"]
ct=ColumnTransformer(transformers=[("one_hot",OneHotEncoder(),categorical_columns)],remainder="passthrough")

stp2=LinearRegression()

pipe=Pipeline([("ct",ct),("stp2",stp2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

np.exp(0.21)

"""KNN"""

step2=KNeighborsRegressor(n_neighbors=3)

pipe=Pipeline([("ct",ct),("step2",step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

step3=DecisionTreeRegressor(max_depth=8)

pipe=Pipeline([("ct",ct),("step3",step3)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

step4=RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=18)

pipe=Pipeline([("ct",ct),("step4",step4)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

step5=SVR()

pipe=Pipeline([("ct",ct),("step5",step5)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

step6=XGBRegressor(n_estimators=100,max_depth=8)

pipe=Pipeline([("ct",ct),("step6",step6)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

"""Exporting the Model"""

import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))