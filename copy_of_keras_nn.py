import numpy as np
import pandas as pd

#  keras 
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

#  sklearn 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('./data/regresion_data.csv')

df = df.drop(columns = 'label')
df = df.drop(columns = 'Unnamed: 0')

X = df.drop(columns = 'frequancy')
y = df['frequancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

