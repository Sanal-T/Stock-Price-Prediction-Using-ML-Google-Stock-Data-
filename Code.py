import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

df=pd.read_csv('GOOG.csv')
df.head()

df.info()