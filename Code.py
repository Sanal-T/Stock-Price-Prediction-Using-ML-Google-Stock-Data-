import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

df=pd.read_csv('GOOG.csv')

df.info()

df.shape
df=df.drop(columns=['symbol','adjClose','adjHigh','adjLow','adjOpen','adjOpen','divCash','splitFactor'],axis=1)
df.head()

df.describe()

import plotly.graph_objects as go
figure=go.Figure(data=[go.Candlestick(x=df['date'],open=df['open'],high=df['high'],low=df['low'],close=df['close'])])
figure.update_layout(title='GOOG',xaxis_rangeslider_visible=False)
figure.show()

df=df.drop(columns=['date'],axis=1)
df.head()

df.duplicated().sum().any()
df.isnull().values.any()
df.describe()

print(df.corr())

plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),cmap='Blues',annot=True)
plt.show()

sns.pairplot(df)

df['open'].hist()

df['close'].hist()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,f1_score

label_encoder = LabelEncoder()
df['open'] = label_encoder.fit_transform(df['open'])
df['high'] = label_encoder.fit_transform(df['high'])
df['low'] = label_encoder.fit_transform(df['low'])
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

x=df[['open','high','low','volume']].values
y=df['close'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)