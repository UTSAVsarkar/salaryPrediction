import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

if __name__ == '__main__':
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    regressor = LinearRegression()
    regressor.fit(X, y)
    file = open('model.pkl', 'wb')
    pickle.dump(regressor, file)
    file.close()